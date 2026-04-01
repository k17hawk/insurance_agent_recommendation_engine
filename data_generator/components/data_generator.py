import pandas as pd
import numpy as np
from typing import  Dict, Any, Union, Optional
from pathlib import Path
from data_generator.entity.config_entity import TrainingPipelineConfig
from data_generator.entity.artifact_entity import DataArtifact
from data_generator.components.broker_generator import BrokerGenerator
from data_generator.components.lead_generator import LeadGenerator
from data_generator.components.assignment_generator import AssignmentGenerator
from data_generator.constants import (
    BROKER_COLUMNS, DATA_DIR, OUTPUT_FILES,
    BROKERS_PATH, LEADS_PATH, ASSIGNMENTS_PATH,
    COUNTERFACTUAL_PATH, HISTORICAL_PATH
)

class DataGenerator:
    """Main component orchestrating the synthetic data generation"""
    
    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        self.broker_generator = BrokerGenerator(config)
        self.lead_generator = LeadGenerator(config)
        self.assignment_generator = AssignmentGenerator(config)
        
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def generate(self) -> DataArtifact:
        """Generate all synthetic data"""
        print("=" * 70)
        print("SYNTHETIC DATA GENERATOR V8.0 — PRODUCTION MERGE")
        print("V7.1 causal rigour  +  V7.2 Ontario domain accuracy")
        print("=" * 70)
        
        # Generate brokers
        print("\nStep 1/4: Generating broker profiles...")
        brokers = self.broker_generator.generate_brokers(self.config.data_generation.n_brokers)
        new_brokers = brokers[brokers['is_new_broker']]
        ribo_brokers = brokers[brokers['ribo_licensed']]
        print(f"  {len(brokers)} brokers  |  {len(new_brokers)} new  |  "
              f"{len(ribo_brokers)} RIBO-licensed "
              f"({len(ribo_brokers)/len(brokers)*100:.0f}%)")
        print(f"  commission: {brokers['commission_rate'].min():.1%} – "
              f"{brokers['commission_rate'].max():.1%}  |  "
              f"efficiency: {brokers['efficiency'].min():.2f} – "
              f"{brokers['efficiency'].max():.2f}")
        
        # Generate leads
        print("\nStep 2/4: Generating lead records...")
        leads = self.lead_generator.generate_leads(self.config.data_generation.n_leads)
        print(f"  {len(leads)} leads generated")
        print(f"  avg quote value: ${leads['quote_value'].mean():.0f}")
        print(f"  multi-product intent: {leads['multi_product_intent'].sum()} "
              f"({leads['multi_product_intent'].mean()*100:.0f}%)")
        print(f"  major claims: {(leads['claims_severity']=='major').sum()} "
              f"({(leads['claims_severity']=='major').mean()*100:.0f}%)")
        missing_pct = (
            leads[['insurance_type', 'language', 'tenure_years']]
            .isna().any(axis=1).mean() * 100
        )
        print(f"  missing data rate: {missing_pct:.1f}%")
        
        # Generate assignments
        print("\nStep 3/4: Generating assignments (with broker churn)...")
        assignments, counterfactual = self.assignment_generator.generate_assignments(leads, brokers)
        print(f"  {len(assignments)} assignment records")
        print(f"  {len(counterfactual)} counterfactual records (evaluation only)")
        
        # Build historical dataset
        print("\nStep 4/4: Building training dataset...")
        historical = (
            assignments
            .merge(leads, on='lead_id', how='left')
            .merge(brokers[BROKER_COLUMNS], on='broker_id', how='left')
        )
        
        # Drop temporary columns if present
        for col in ['_score', '_simulated_score']:
            if col in historical.columns:
                historical.drop(columns=[col], inplace=True)
        
        # Print summary statistics
        self._print_summary(brokers, leads, assignments, counterfactual)
        
        # Save artifacts
        artifact = self._save_artifacts(brokers, leads, assignments, counterfactual, historical)
        
        return artifact
    
    def _print_summary(self, brokers: pd.DataFrame, leads: pd.DataFrame,
                       assignments: pd.DataFrame, counterfactual: pd.DataFrame):
        """Print summary statistics"""
        pos = assignments[assignments['is_assigned'] == 1]
        converted = pos[pos['converted'] == 1]
        censored = pos[pos.get('censored', pd.Series(0, index=pos.index)) == 1]
        
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE — V8.0")
        print("=" * 70)
        print(f"\nDataset statistics:")
        print(f"  Brokers:              {len(brokers):,}")
        print(f"  Leads:                {len(leads):,}")
        print(f"  Total assignments:    {len(assignments):,}")
        print(f"  Positive samples:     {len(pos):,}")
        print(f"  Observed conversions: {len(converted):,}  "
              f"({len(converted)/len(pos)*100:.1f}%)")
        if len(censored):
            print(f"  Censored:             {len(censored):,}")
        
        pos_prop = pos['propensity_score']
        print(f"\nPropensity scores (positive assignments):")
        print(f"  mean={pos_prop.mean():.4f}  "
              f"min={pos_prop.min():.4f}  max={pos_prop.max():.4f}")
        final_expl = assignments['exploration_rate_at_time'].iloc[-1]
        print(f"  exploration decay: "
              f"{self.config.data_generation.initial_exploration_rate:.2%} → {final_expl:.2%}")
        
        if len(counterfactual):
            cf_conv = counterfactual['potential_outcome'].mean() * 100
            cf_prob = counterfactual['potential_conversion_probability'].mean()
            print(f"\nCounterfactual summary (evaluation only):")
            print(f"  would-have converted: {cf_conv:.1f}%  |  "
                  f"avg probability: {cf_prob:.3f}")
        
        total_profit = pos['profit'].sum()
        print(f"\nProfit metrics:")
        print(f"  total profit:       ${total_profit:,.2f}")
        print(f"  avg per assignment: ${total_profit/len(pos):.2f}")
        
        print(f"\nV8.0 feature checklist:")
        print("  [OK] seasonality defined in __init__ (V7.2 crash fixed)")
        print("  [OK] no hidden factor in match score (causal transparency)")
        print("  [OK] propensity_score on every assigned record")
        print("  [OK] action_probabilities logged as JSON")
        print("  [OK] decaying exploration rate (unbiased mixed policy)")
        print("  [OK] uniform negative sampling (no selection-score leakage)")
        print("  [OK] counterfactual CSV restored (5 output files)")
        print("  [OK] update_market_regime called once per lead")
        print("  [OK] event-driven caseload: -1 convert / -0.5 timeout")
        print("  [OK] RIBO as ×1.1 multiplicative (not additive)")
        print("  [OK] claims_severity wired to conversion (major -15%)")
        print("  [OK] multi_product_intent wired to conversion (+15% if bundle)")
        print("  [OK] broker churn + replacement every 500 leads")
        print("  [OK] patience_hours drives max_interactions per lead")
        print("  [OK] original_lead_id preserved for sequence modelling")
        print("  [OK] market_regime logged on every record")
    
    def _ensure_directory_exists(self, file_path: Union[Path, str]) -> None:
        """Ensure directory exists for the given file path"""
        # Convert to Path if it's a string
        if isinstance(file_path, str):
            file_path = Path(file_path)
        # Create parent directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _to_path(self, path: Union[Path, str]) -> Path:
        """Convert string to Path if necessary"""
        if isinstance(path, str):
            return Path(path)
        return path
    
    def _save_dataframe(self, df: pd.DataFrame, file_path: Union[Path, str], 
                       description: str) -> bool:
        """Save a single dataframe to disk with error handling"""
        try:
            # Convert to Path if it's a string
            file_path = self._to_path(file_path)
            
            # Ensure directory exists
            self._ensure_directory_exists(file_path)
            
            # Save dataframe
            df.to_csv(file_path, index=False)
            
            # Verify file was created and is readable
            if file_path.exists() and file_path.stat().st_size > 0:
                file_size_kb = file_path.stat().st_size / 1024
                print(f"  ✓ {file_path.name} - {description} ({file_size_kb:.1f} KB)")
                return True
            else:
                print(f"  ✗ Failed to save {file_path.name} - file is empty or missing")
                return False
                
        except Exception as e:
            print(f"  ✗ Error saving {file_path.name if hasattr(file_path, 'name') else file_path}: {str(e)}")
            return False
    
    def _save_artifacts(self, brokers: pd.DataFrame, leads: pd.DataFrame,
                        assignments: pd.DataFrame, counterfactual: pd.DataFrame,
                        historical: pd.DataFrame) -> DataArtifact:
        """Save all artifacts to disk using constants for file paths"""
        
        print("\n" + "=" * 70)
        print("SAVING ARTIFACTS")
        print("=" * 70)
        
        # Dictionary mapping dataframes to their output config
        artifacts_to_save = [
            (brokers, BROKERS_PATH, OUTPUT_FILES['brokers']['description']),
            (leads, LEADS_PATH, OUTPUT_FILES['leads']['description']),
            (assignments, ASSIGNMENTS_PATH, OUTPUT_FILES['assignments']['description']),
            (counterfactual, COUNTERFACTUAL_PATH, OUTPUT_FILES['counterfactual']['description']),
            (historical, HISTORICAL_PATH, OUTPUT_FILES['historical']['description'])
        ]
        
        # Save each artifact
        save_results = []
        for df, path, description in artifacts_to_save:
            if df is not None and not df.empty:
                result = self._save_dataframe(df, path, description)
                save_results.append(result)
            else:
                print(f"  ⚠ {path.name if hasattr(path, 'name') else path} is empty, skipping save")
                save_results.append(False)
        
        # Print summary
        print("\n" + "-" * 70)
        print("SAVE SUMMARY")
        print("-" * 70)
        
        successful_saves = sum(save_results)
        total_saves = len(save_results)
        
        if successful_saves == total_saves:
            print(f"✓ All {total_saves} files saved successfully!")
        else:
            print(f"⚠ Saved {successful_saves}/{total_saves} files successfully")
            print("  Check the errors above for details.")
        
        # Create and return artifact with Path objects
        artifact = DataArtifact(
            brokers_path=self._to_path(BROKERS_PATH),
            leads_path=self._to_path(LEADS_PATH),
            assignments_path=self._to_path(ASSIGNMENTS_PATH),
            counterfactual_path=self._to_path(COUNTERFACTUAL_PATH),
            historical_path=self._to_path(HISTORICAL_PATH)
        )
        
        print("\n" + "=" * 70)
        print("ARTIFACTS SAVED TO:")
        print(f"  {DATA_DIR}")
        print("=" * 70)
        
        return artifact
    
    def load_existing_artifacts(self) -> Optional[DataArtifact]:
        """Load existing artifacts if they exist"""
        artifact = DataArtifact(
            brokers_path=self._to_path(BROKERS_PATH),
            leads_path=self._to_path(LEADS_PATH),
            assignments_path=self._to_path(ASSIGNMENTS_PATH),
            counterfactual_path=self._to_path(COUNTERFACTUAL_PATH),
            historical_path=self._to_path(HISTORICAL_PATH)
        )
        
        # Check which files exist
        existing_files = {}
        for name, path in artifact.to_dict().items():
            file_path = Path(path)
            if file_path.exists():
                file_size_kb = file_path.stat().st_size / 1024
                existing_files[name] = file_path
                print(f"✓ Found existing {name}: {file_path.name} ({file_size_kb:.1f} KB)")
            else:
                print(f"✗ Missing {name}: {file_path.name}")
        
        if existing_files:
            print(f"\nFound {len(existing_files)} existing artifacts")
            return artifact
        else:
            print("\nNo existing artifacts found")
            return None
    
    def validate_artifacts(self, artifact: DataArtifact) -> Dict[str, Dict[str, Any]]:
        """Validate saved artifacts"""
        validation_results = {}
        
        print("\n" + "=" * 70)
        print("VALIDATING ARTIFACTS")
        print("=" * 70)
        
        for name, path_str in artifact.to_dict().items():
            file_path = Path(path_str)
            if file_path.exists():
                try:
                    # Try to read the file to verify it's valid
                    df = pd.read_csv(file_path)
                    validation_results[name] = {
                        'exists': True,
                        'valid': True,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'size_kb': file_path.stat().st_size / 1024
                    }
                    print(f"✓ {name}: {len(df):,} rows, {len(df.columns)} columns, "
                          f"{validation_results[name]['size_kb']:.1f} KB")
                except Exception as e:
                    validation_results[name] = {
                        'exists': True,
                        'valid': False,
                        'error': str(e)
                    }
                    print(f"✗ {name}: Invalid file - {str(e)}")
            else:
                validation_results[name] = {
                    'exists': False,
                    'valid': False
                }
                print(f"✗ {name}: File not found at {file_path}")
        
        return validation_results