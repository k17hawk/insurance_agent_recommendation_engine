import numpy as np
import pandas as pd
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from data_generator.entity.config_entity import TrainingPipelineConfig



class AssignmentGenerator:
    """Component for generating assignment records with simulated conversions"""
    
    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        self.global_time_step = 0
        self.market_regime = config.market_regime_config.initial_regime
        self.regime_counter = 0
        self.regime_durations = config.market_regime_config.regime_durations
        self.regime_transition = config.market_regime_config.regime_transition
        self.regime_effects = config.market_regime_config.regime_effects
    
    def get_current_exploration_rate(self) -> float:
        """Calculate decaying exploration rate"""
        return max(0.05,
                  self.config.data_generation.initial_exploration_rate *
                  np.exp(-self.global_time_step * self.config.data_generation.exploration_decay))
    
    def update_market_regime(self):
        """Update market regime based on Markov transition"""
        self.regime_counter += 1
        if self.regime_counter >= self.regime_durations[self.market_regime]:
            self.regime_counter = 0
            probs = self.regime_transition[self.market_regime]
            self.market_regime = np.random.choice(list(probs.keys()), p=list(probs.values()))
    
    def get_market_factor(self) -> float:
        """Get current market factor"""
        return self.regime_effects[self.market_regime]
    
    def calculate_match_score(self, lead: Dict, broker: Dict) -> float:
        """Calculate match score between lead and broker"""
        # Expertise
        insurance_type = lead.get('insurance_type')
        expertise = 0.0
        if pd.notna(insurance_type):
            if insurance_type == 'auto' and broker.get('expertise_auto'):
                expertise = 1.0
            elif insurance_type == 'home' and broker.get('expertise_home'):
                expertise = 1.0
            elif insurance_type == 'bundle':
                if broker.get('expertise_bundle'):
                    expertise = 1.0
                elif broker.get('expertise_auto') and broker.get('expertise_home'):
                    expertise = 0.9
                else:
                    expertise = 0.6
        
        # Language
        lang = lead.get('language')
        if pd.notna(lang):
            language = 1.0 if broker.get('languages') in ('Bilingual', lang) else 0.3
        else:
            language = 0.8
        
        # Workload
        workload = np.clip(1 - broker.get('current_caseload', 0) / broker.get('capacity', 40), 0.2, 1.0)
        
        # Quality
        quality = np.clip(broker.get('conversion_rate', 0.1) * 0.6 + (broker.get('csat_score', 3) / 5) * 0.4, 0, 1)
        
        # Response time
        response = max(0.5, min(1.0, 1 - broker.get('avg_response_time', 8) / 24))
        
        observed_score = (expertise + language + workload + quality + response) / 5
        
        # RIBO boost
        if broker.get('ribo_licensed', False):
            observed_score = min(1.0, observed_score * 1.1)
        
        return np.clip(observed_score, 0.1, 1.0)
    
    def calculate_price_sensitivity(self, quote_value: float, insurance_type: str, lead: Dict) -> float:
        """Calculate price sensitivity factor"""
        expected = self.config.lead_config.expected_premium_base.get(insurance_type,
                   self.config.lead_config.expected_premium_base['auto'])
        if lead.get('tenure_years', 0) and lead['tenure_years'] > 0:
            expected *= 0.95
        price_ratio = quote_value / expected
        price_effect = 1 / (1 + np.exp(self.config.conversion_config.price_sensitivity_k * (price_ratio - 1)))
        return np.clip(price_effect, 0.30, 1.20)
    
    def sigmoid_conversion(self, base_score: float, lead: Dict, broker: Dict,
                          workload_ratio: float = 0.5, quote_value: float = None) -> float:
        """Calculate conversion probability using sigmoid"""
        adjusted = base_score
        adjusted *= (1 - workload_ratio * 0.3)
        adjusted *= lead.get('lead_difficulty', 1.0)
        
        # Price sensitivity
        ins_type = lead.get('insurance_type', 'auto')
        if pd.notna(ins_type) and quote_value is not None:
            adjusted *= self.calculate_price_sensitivity(quote_value, ins_type, lead)
        
        # Claims history
        claims_severity = lead.get('claims_severity', 'none')
        if claims_severity == 'major':
            adjusted *= self.config.conversion_config.claims_factors['major']
        elif claims_severity == 'minor':
            adjusted *= self.config.conversion_config.claims_factors['minor']
        
        # Multi-product intent
        if lead.get('multi_product_intent', False):
            if broker.get('expertise_bundle', 0):
                adjusted *= self.config.conversion_config.multi_product_factors['bundle_expertise']
            elif broker.get('expertise_auto', 0) and broker.get('expertise_home', 0):
                adjusted *= self.config.conversion_config.multi_product_factors['auto_home_expertise']
        
        raw_prob = 1 / (1 + np.exp(-self.config.conversion_config.sigmoid_k *
                                  (adjusted - self.config.conversion_config.threshold)))
        return np.clip(raw_prob * 0.5,
                      self.config.conversion_config.conversion_prob_range[0],
                      self.config.conversion_config.conversion_prob_range[1])
    
    def handle_broker_churn(self, broker_state: Dict, brokers_df: pd.DataFrame) -> Tuple[Dict, List, List]:
        """Handle broker churn and replacement"""
        churned = []
        replacements = []
        
        for broker_id, state in list(broker_state.items()):
            if not state.get('active', True):
                continue
            row = brokers_df[brokers_df['broker_id'] == broker_id]
            if row.empty:
                continue
            broker = row.iloc[0]
            
            churn_prob = self.config.churn_config.baseline_prob
            if state['skill'] < 0.30:
                churn_prob += self.config.churn_config.low_skill_penalty
            if state['caseload'] > broker['capacity'] * self.config.churn_config.capacity_overflow_threshold:
                churn_prob += self.config.churn_config.overcapacity_penalty
            if broker.get('years_experience', 0) > 30:
                churn_prob += self.config.churn_config.senior_penalty
            if state.get('burnout_risk', 0) > self.config.churn_config.burnout_threshold:
                churn_prob += self.config.churn_config.high_burnout_penalty
            
            if random.random() < churn_prob:
                broker_state[broker_id]['active'] = False
                churned.append(broker_id)
                
                new_broker = self.create_replacement_broker(brokers_df)
                broker_state[new_broker['broker_id']] = {
                    'caseload': new_broker['current_caseload'],
                    'skill': new_broker['skill_level'],
                    'recent_success': 0.5,
                    'burnout_risk': new_broker['burnout_risk'],
                    'active': True
                }
                replacements.append(new_broker)
        
        return broker_state, churned, replacements
    
    def create_replacement_broker(self, brokers_df: pd.DataFrame) -> Dict:
        """Create a replacement broker for churned ones"""
        max_num = max(int(b.split('-')[1]) for b in brokers_df['broker_id'])
        new_id = f'BR-{max_num + 1:04d}'
        region = np.random.choice(list(self.config.region_config.regions.keys()),
                                 p=list(self.config.region_config.regions.values()))
        
        n_expertise = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        expertise = random.sample(self.config.broker_config.expertise_areas, n_expertise)
        lang = np.random.choice(['English', 'French', 'Bilingual'], p=[0.85, 0.05, 0.10])
        
        ribo_years = np.random.randint(0, 3)
        ribo_licensed = ribo_years >= 1
        
        skill = np.clip(np.random.normal(
            self.config.churn_config.replacement_broker_skill['mean'],
            self.config.churn_config.replacement_broker_skill['std']),
            self.config.churn_config.replacement_broker_skill['range'][0],
            self.config.churn_config.replacement_broker_skill['range'][1]
        )
        
        return {
            'broker_id': new_id,
            'region': region,
            'expertise_auto': int('auto' in expertise),
            'expertise_home': int('home' in expertise),
            'expertise_commercial': int('commercial' in expertise),
            'expertise_bundle': int('bundle' in expertise),
            'languages': lang,
            'ribo_licensed': ribo_licensed,
            'ribo_license_years': ribo_years,
            'conversion_rate': round(np.clip(skill * 0.4 + np.random.normal(0, 0.05), 0.05, 0.55), 3),
            'csat_score': round(np.clip(np.random.normal(3.8 + skill * 0.5, 0.5), 2.5, 4.8), 2),
            'current_caseload': max(3, np.random.poisson(10)),
            'capacity': np.random.choice([40, 50, 60], p=[0.5, 0.3, 0.2]),
            'avg_response_time': round(np.random.uniform(3, 12), 1),
            'skill_level': round(skill, 3),
            'learning_rate': round(np.random.uniform(0.10, 0.30), 3),
            'burnout_risk': round(np.random.uniform(0.0, 0.20), 3),
            'reliability': round(np.random.uniform(0.75, 0.95), 3),
            'commission_rate': round(np.random.uniform(0.08, 0.15), 3),
            'cost_per_lead': round(np.random.uniform(50, 150), 2),
            'efficiency': round(np.random.uniform(0.5, 1.5), 3),
            'years_experience': 0,
            'is_new_broker': True,
            'active': True
        }
    
    def get_eligible_brokers(self, lead: Dict, brokers_df: pd.DataFrame) -> pd.DataFrame:
        """Get eligible brokers for a lead"""
        same_region = brokers_df[brokers_df['region'] == lead['region']]
        nearby = pd.DataFrame()
        if lead['region'] in self.config.region_config.nearby_regions and random.random() < 0.3:
            nearby = brokers_df[brokers_df['region'].isin(
                self.config.region_config.nearby_regions[lead['region']]
            )]
        
        if len(same_region) < 5:
            eligible = pd.concat([same_region, nearby])
        elif random.random() < 0.2 and len(nearby) > 0:
            eligible = pd.concat([same_region.sample(min(3, len(same_region))), nearby])
        else:
            eligible = same_region
        
        eligible = eligible[eligible['current_caseload'] <= eligible['capacity']]
        if len(eligible) == 0:
            eligible = brokers_df
        return eligible
    
    def simulate_journey(self, lead: Dict, brokers: pd.DataFrame,
                        broker_state: Dict, depth: int = 0) -> Tuple[List, List]:
        """Simulate the journey for a single lead"""
        assignments = []
        counterfactual_samples = []
        attempted = []
        current_exploration = self.get_current_exploration_rate()
        
        patience_hours = lead.get('patience_hours', 48)
        max_interactions = min(self.config.assignment_config.max_interactions_per_lead,
                              int(patience_hours / 24) + 1)
        
        for interaction_num in range(max_interactions):
            # Refresh broker caseloads
            brokers_fresh = brokers.copy()
            brokers_fresh['current_caseload'] = brokers_fresh['broker_id'].map(
                lambda x: broker_state[x]['caseload']
            )
            
            available = brokers_fresh[~brokers_fresh['broker_id'].isin(attempted)]
            if available.empty:
                break
            
            # Score candidates
            scores = []
            for _, b in available.iterrows():
                b_copy = b.copy()
                state = broker_state[b['broker_id']]
                success_sig = state.get('recent_success', 0.5)
                skill_gain = b['learning_rate'] * (1 - state['skill']) * success_sig
                b_copy['skill_level'] = min(0.95, state['skill'] + skill_gain)
                burnout = state.get('burnout_risk', 0)
                b_copy['skill_level'] *= (1 - burnout * 0.5)
                scores.append(self.calculate_match_score(lead, b_copy))
            
            available = available.copy()
            available['_score'] = scores
            available = available.sort_values('_score', ascending=False).head(10)
            
            # Softmax selection
            logits = np.exp(available['_score'] - available['_score'].max())
            probs = logits / logits.sum()
            n_cand = len(available)
            
            action_probs = {str(bid): float(p) for bid, p in zip(available['broker_id'], probs)}
            
            # Mixed explore/exploit
            if random.random() < current_exploration:
                selected = available.sample(1).iloc[0]
                sel_idx = available.index.get_loc(selected.name)
                true_prop = (current_exploration * (1 / n_cand) + (1 - current_exploration) * probs.iloc[sel_idx])
            else:
                sel_idx = np.random.choice(len(available), p=probs)
                selected = available.iloc[sel_idx]
                true_prop = (current_exploration * (1 / n_cand) + (1 - current_exploration) * probs.iloc[sel_idx])
            
            attempted.append(selected['broker_id'])
            workload_ratio = selected['current_caseload'] / selected['capacity']
            
            # Update burnout
            broker_state[selected['broker_id']]['burnout_risk'] = min(0.80,
                broker_state[selected['broker_id']].get('burnout_risk', 0) + workload_ratio * 0.05
            )
            
            # Response check
            burnout_now = broker_state[selected['broker_id']].get('burnout_risk', 0)
            eff_reliability = selected['reliability'] * (1 - burnout_now * 0.5) * selected['efficiency']
            
            if random.random() > eff_reliability:
                assignments.append({
                    'lead_id': lead['lead_id'],
                    'original_lead_id': lead.get('original_lead_id', lead['lead_id']),
                    'broker_id': selected['broker_id'],
                    'assignment_date': lead['lead_date'],
                    'converted': 0,
                    'censored': 0,
                    'conversion_delay_days': np.nan,
                    'conversion_value': 0,
                    'revenue': 0,
                    'cost': selected['cost_per_lead'],
                    'profit': -selected['cost_per_lead'],
                    'expected_profit': -selected['cost_per_lead'],
                    'is_assigned': 1,
                    'interaction_number': interaction_num,
                    'responded': 0,
                    'response_time_hours': np.nan,
                    'propensity_score': round(float(true_prop), 6),
                    'exploration_rate_at_time': round(current_exploration, 4),
                    'action_probabilities': json.dumps(action_probs),
                    'position_bias': 0.0,
                    'market_regime': self.market_regime,
                })
                continue
            
            # Response time
            queue_mult = 1 + workload_ratio * np.random.uniform(0.5, 1.5) / selected['efficiency']
            response_time = np.clip(
                np.random.exponential(selected['avg_response_time']) * queue_mult,
                0.5, 72
            )
            
            position_bias = float(np.exp(-0.3 * interaction_num))
            
            # Competition effect
            top_scores = available['_score'].head(3).values
            if len(top_scores) >= 2:
                competition_factor = 1 - np.clip(top_scores[0] - top_scores[1], 0, 0.3)
            else:
                competition_factor = 1.0
            
            # Conversion probability
            base_score = self.calculate_match_score(lead, selected)
            conversion_prob = self.sigmoid_conversion(
                base_score, lead, selected, workload_ratio, lead['quote_value']
            )
            conversion_prob *= position_bias
            conversion_prob *= competition_factor
            conversion_prob *= self.config.lead_config.seasonality.get(lead.get('month', 6), 1.0)
            conversion_prob *= self.get_market_factor()
            conversion_prob = np.clip(conversion_prob,
                                     self.config.conversion_config.conversion_prob_range[0],
                                     self.config.conversion_config.conversion_prob_range[1])
            
            will_convert = random.random() < conversion_prob
            
            revenue = lead['quote_value'] * selected['commission_rate'] if will_convert else 0
            cost = selected['cost_per_lead']
            profit = revenue - cost if will_convert else -cost
            expected_profit = (lead['quote_value'] * selected['commission_rate'] * conversion_prob) - cost
            
            common = {
                'lead_id': lead['lead_id'],
                'original_lead_id': lead.get('original_lead_id', lead['lead_id']),
                'broker_id': selected['broker_id'],
                'assignment_date': lead['lead_date'],
                'censored': 0,
                'conversion_delay_days': np.nan,
                'conversion_value': 0,
                'revenue': 0,
                'cost': cost,
                'profit': -cost,
                'expected_profit': round(expected_profit, 2),
                'is_assigned': 1,
                'interaction_number': interaction_num,
                'responded': 1,
                'response_time_hours': round(response_time, 1),
                'propensity_score': round(float(true_prop), 6),
                'exploration_rate_at_time': round(current_exploration, 4),
                'action_probabilities': json.dumps(action_probs),
                'position_bias': round(position_bias, 4),
                'market_regime': self.market_regime,
            }
            
            if will_convert:
                delay = np.clip(
                    np.random.exponential(self.config.assignment_config.response_time_scale / selected['efficiency']),
                    0.1, 30
                )
                observed = delay <= self.config.data_generation.observation_window_days
                
                rec = {**common,
                       'converted': int(observed),
                       'censored': 1 if (will_convert and not observed) else 0,
                       'conversion_delay_days': round(delay, 1) if observed else np.nan,
                       'conversion_value': lead['quote_value'] if observed else 0,
                       'revenue': revenue if observed else 0,
                       'profit': profit if observed else -cost}
                assignments.append(rec)
                
                # Update skill
                success_sig = conversion_prob / self.config.conversion_config.conversion_prob_range[1]
                state = broker_state[selected['broker_id']]
                state['skill'] = min(0.95,
                    state['skill'] + selected['learning_rate'] * (1 - state['skill']) * success_sig
                )
                state['caseload'] = max(0, state['caseload'] - 1)
                state['recent_success'] = success_sig
                state['burnout_risk'] = max(0, state.get('burnout_risk', 0) - 0.10)
                break
            else:
                assignments.append({**common, 'converted': 0})
                state = broker_state[selected['broker_id']]
                state['caseload'] = max(0, state['caseload'] - 0.5)
                state['recent_success'] = max(0.2, state.get('recent_success', 0.5) * 0.95)
        
        # Uniform negative sampling
        n_negatives = min(3, len(brokers) - len(attempted))
        if n_negatives > 0:
            unassigned = brokers[~brokers['broker_id'].isin(attempted)]
            if len(unassigned) > n_negatives:
                negatives = unassigned.sample(n=n_negatives)
            else:
                negatives = unassigned
            
            for _, neg in negatives.iterrows():
                would_prob = self.sigmoid_conversion(
                    self.calculate_match_score(lead, neg),
                    lead, neg, 0.5, lead['quote_value']
                )
                soft_convert = random.random() < would_prob
                cf_revenue = lead['quote_value'] * neg['commission_rate'] if soft_convert else 0
                cf_exp_profit = lead['quote_value'] * neg['commission_rate'] * would_prob
                
                assignments.append({
                    'lead_id': lead['lead_id'],
                    'original_lead_id': lead.get('original_lead_id', lead['lead_id']),
                    'broker_id': neg['broker_id'],
                    'assignment_date': lead['lead_date'],
                    'converted': 0,
                    'censored': 0,
                    'conversion_delay_days': np.nan,
                    'conversion_value': 0,
                    'revenue': 0,
                    'cost': 0,
                    'profit': 0,
                    'expected_profit': round(cf_exp_profit, 2),
                    'is_assigned': 0,
                    'interaction_number': -1,
                    'responded': np.nan,
                    'response_time_hours': np.nan,
                    'propensity_score': 0.0,
                    'exploration_rate_at_time': round(current_exploration, 4),
                    'action_probabilities': '{}',
                    'position_bias': 0.0,
                    'market_regime': self.market_regime,
                })
                
                counterfactual_samples.append({
                    'lead_id': lead['lead_id'],
                    'original_lead_id': lead.get('original_lead_id', lead['lead_id']),
                    'broker_id': neg['broker_id'],
                    'potential_outcome': int(soft_convert),
                    'potential_conversion_probability': round(would_prob, 4),
                    'potential_revenue': cf_revenue,
                    'potential_profit': cf_revenue,
                    'match_quality': self.calculate_match_score(lead, neg),
                })
        
        return assignments, counterfactual_samples
    
    def generate_assignments(self, leads: pd.DataFrame, brokers: pd.DataFrame,
                           depth: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate assignments for all leads"""
        broker_state = {
            row['broker_id']: {
                'skill': row['skill_level'],
                'caseload': row['current_caseload'],
                'recent_success': 0.5,
                'burnout_risk': row.get('burnout_risk', 0.0),
                'active': True,
            }
            for _, row in brokers.iterrows()
        }
        
        replacement_rows = []
        assignments = []
        all_counterfactual = []
        leads_with_reentry = []
        
        print(f"  Processing {len(leads)} leads (depth {depth})...")
        
        for idx, lead in leads.iterrows():
            if idx % 2000 == 0 and idx > 0:
                print(f"    ... {idx}/{len(leads)} leads processed")
            
            self.global_time_step += 1
            self.update_market_regime()
            
            # Burnout recovery
            for k, state in broker_state.items():
                if state.get('active', True):
                    state['burnout_risk'] = max(0.0, state.get('burnout_risk', 0) - 0.001)
            
            # Broker churn
            if idx > 0 and idx % self.config.churn_config.churn_frequency == 0:
                broker_state, churned, replacements = self.handle_broker_churn(broker_state, brokers)
                if churned:
                    print(f"    Churn at lead {idx}: {len(churned)} retired, {len(replacements)} hired")
                for rep in replacements:
                    replacement_rows.append(rep)
                    brokers = pd.concat([brokers, pd.DataFrame([rep])], ignore_index=True)
            
            # Build live snapshot
            brokers_dynamic = brokers.copy()
            brokers_dynamic['current_caseload'] = brokers_dynamic['broker_id'].map(
                lambda x: broker_state[x]['caseload'] if broker_state.get(x, {}).get('active', True) else 9999
            )
            active_brokers = brokers_dynamic[brokers_dynamic['current_caseload'] < 9999]
            
            eligible = self.get_eligible_brokers(lead, active_brokers)
            if eligible.empty:
                continue
            
            lead_assignments, lead_cf = self.simulate_journey(lead, eligible, broker_state, depth)
            
            for rec in lead_assignments:
                if random.random() < self.config.assignment_config.missing_data_log_rate:
                    continue
                if random.random() < self.config.assignment_config.timestamp_jitter_rate:
                    noise = np.random.randint(-self.config.assignment_config.timestamp_jitter_hours,
                                            self.config.assignment_config.timestamp_jitter_hours)
                    if isinstance(rec.get('assignment_date'), datetime):
                        rec['assignment_date'] += timedelta(hours=noise)
                assignments.append(rec)
            
            all_counterfactual.extend(lead_cf)
            
            # Lead re-entry
            if random.random() < self.config.assignment_config.reentry_rate and depth < self.config.assignment_config.max_reentry_depth:
                reentry = lead.copy()
                reentry['lead_date'] = lead['lead_date'] + timedelta(days=np.random.randint(7, 30))
                reentry['lead_id'] = f"{lead['lead_id']}_R{len(leads_with_reentry)}"
                reentry['original_lead_id'] = lead['lead_id']
                leads_with_reentry.append(reentry)
        
        # Process re-entered leads
        if leads_with_reentry and depth < self.config.assignment_config.max_reentry_depth:
            print(f"  Processing {len(leads_with_reentry)} re-entered leads (depth {depth+1})...")
            reentry_df = pd.DataFrame(leads_with_reentry)
            re_asgn, re_cf = self.generate_assignments(reentry_df, brokers, depth + 1)
            assignments.extend(re_asgn.to_dict('records') if isinstance(re_asgn, pd.DataFrame) else re_asgn)
            all_counterfactual.extend(re_cf.to_dict('records') if isinstance(re_cf, pd.DataFrame) else re_cf)
        
        return pd.DataFrame(assignments), pd.DataFrame(all_counterfactual)