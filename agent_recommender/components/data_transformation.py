import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import json
from agent_recommender import logger
from agent_recommender.entity.config_entity import DataTransformationConfig
from agent_recommender.utils.utility import create_directories


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.df = None
        self.df_positive = None
        self.df_negative = None
        self.train_df = None
        self.test_df = None
        self.leads_full = None
        self.brokers_full = None
        self.counterfactual_clean = None
        self.assignments_clean = None
        self.SEED = 42
        
        # Store fitted transformers
        self.label_encoders = {}
        self.median_values = {}
        self.quality_scales = {}
        self.outlier_bounds = {}
        self.max_quote_value = None
        
        # Define feature lists (will be filled after saving)
        self.client_features = None
        self.broker_features = None
        self.interaction_features = None
        
    def load_data(self):
        """Load preprocessed data from data_ingestion stage"""
        logger.info("Loading preprocessed data...")
        
        preprocessed_dir = self.config.preprocessed_dir
        
        # Load all required files
        self.leads_full = pd.read_csv(preprocessed_dir / "leads_full.csv")
        self.brokers_full = pd.read_csv(preprocessed_dir / "brokers_full.csv")
        self.assignments_clean = pd.read_csv(preprocessed_dir / "assignments_clean.csv")
        self.counterfactual_clean = pd.read_csv(preprocessed_dir / "counterfactual_clean.csv")
        
        logger.info(f"Loaded leads_full: {len(self.leads_full):,} rows")
        logger.info(f"Loaded brokers_full: {len(self.brokers_full):,} rows")
        logger.info(f"Loaded assignments_clean: {len(self.assignments_clean):,} rows")
        logger.info(f"Loaded counterfactual_clean: {len(self.counterfactual_clean):,} rows")
        
        return self
    
    def merge_historical_data(self):
        """Merge assignments with leads and brokers to create historical dataset"""
        logger.info("Merging historical data...")
        
        # Define broker columns to merge
        broker_cols_to_merge = [
            "broker_id", "region", "expertise_auto", "expertise_home",
            "expertise_bundle", "conversion_rate", "csat_score", "languages",
            "ribo_licensed", "ribo_license_years", "capacity", "avg_response_time",
            "is_new_broker", "skill_level", "reliability", "commission_rate",
            "cost_per_lead", "efficiency", "burnout_risk", "utilization", "is_overloaded"
        ]
        broker_cols_to_merge = [c for c in broker_cols_to_merge if c in self.brokers_full.columns]
        
        # Merge all data
        self.df = (
            self.assignments_clean
            .merge(self.leads_full, on="lead_id", how="inner")
            .merge(self.brokers_full[broker_cols_to_merge], on="broker_id", how="inner")
        )
        
        # Drop unnecessary columns from notebook
        for col in ["_score", "_simulated_score"]:
            if col in self.df.columns:
                self.df.drop(columns=[col], inplace=True)
        
        logger.info(f"Historical merged shape: {self.df.shape}")
        return self
    
    def split_data(self):
        """Split the raw merged data BEFORE any transformations"""
        logger.info("Splitting data into train and test sets BEFORE transformations...")
        
        # Perform stratified train-test split on converted column
        self.train_df, self.test_df = train_test_split(
            self.df, 
            test_size=0.30, 
            random_state=self.SEED, 
            stratify=self.df["converted"]
        )
        
        logger.info(f"Train set size: {len(self.train_df):,} rows ({len(self.train_df)/len(self.df)*100:.1f}%)")
        logger.info(f"Test set size: {len(self.test_df):,} rows ({len(self.test_df)/len(self.df)*100:.1f}%)")
        logger.info(f"Train conversion rate: {self.train_df['converted'].mean()*100:.2f}%")
        logger.info(f"Test conversion rate: {self.test_df['converted'].mean()*100:.2f}%")
        
        return self
    
    def create_missing_flags(self, df):
        """Create informative missing flags"""
        informative_missing_cols = ["insurance_type", "language", "tenure_years", "digital_engagement_score"]
        
        for col in informative_missing_cols:
            if col in df.columns:
                flag_col = f"{col}_missing"
                df[flag_col] = df[col].isna().astype(int)
        
        return df
    
    def impute_categorical(self, df):
        """Impute categorical variables (same for train and test)"""
        categorical_impute = {
            "insurance_type": "UNKNOWN",
            "language": "UNKNOWN",
            "claims_severity": "none",
        }
        
        for col, fill_val in categorical_impute.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)
        
        return df
    
    def impute_numeric(self, df, fit=False):
        """Impute numeric variables with median"""
        numeric_impute_cols = [
            "tenure_years", "digital_engagement_score", 
            "lead_difficulty", "sophistication", "patience_hours"
        ]
        
        for col in numeric_impute_cols:
            if col in df.columns:
                if fit:
                    # Calculate and store median from training data
                    self.median_values[col] = df[col].median()
                    df[col] = df[col].fillna(self.median_values[col])
                    logger.info(f"Fit median for {col}: {self.median_values[col]:.3f}")
                else:
                    # Use stored median for test data
                    if col in self.median_values:
                        df[col] = df[col].fillna(self.median_values[col])
        
        return df
    
    def create_derived_features(self, df):
        """Create derived features like net_margin, expected_roi, etc."""
        # Handle multi_product_intent
        if "multi_product_intent" in df.columns:
            df["multi_product_intent"] = df["multi_product_intent"].fillna(0).astype(int)
        
        # Create net margin
        if "revenue" in df.columns and "cost" in df.columns:
            df["net_margin"] = np.where(
                df["revenue"] > 0,
                (df["revenue"] - df["cost"]) / df["revenue"].replace(0, np.nan),
                np.nan,
            )
            df["net_margin"] = df["net_margin"].fillna(0).round(4)
        
        # Create expected ROI
        if "quote_value" in df.columns and "cost" in df.columns:
            df["expected_roi"] = (
                (df["quote_value"] * df.get("commission_rate", 0.10) - df["cost"])
                / df["cost"].replace(0, np.nan)
            ).round(4)
            df["expected_roi"] = df["expected_roi"].fillna(0)
        
        return df
    
    def create_response_time_buckets(self, df):
        """Create response time buckets"""
        if "response_time_hours" in df.columns:
            df["response_time_bucket"] = pd.cut(
                df["response_time_hours"],
                bins=[-np.inf, 2, 6, 12, 24, 48, np.inf],
                labels=["<2h", "2-6h", "6-12h", "12-24h", "24-48h", ">48h"],
            )
            rt_order = {"<2h": 0, "2-6h": 1, "6-12h": 2, "12-24h": 3, "24-48h": 4, ">48h": 5}
            df["response_time_bucket_ord"] = (
                df["response_time_bucket"].astype(str).map(rt_order).fillna(-1).astype(int)
            )
        
        return df
    
    def create_workload_ratio(self, df):
        """Create workload ratio from utilization"""
        if "utilization" in df.columns:
            df["workload_ratio"] = np.clip(df["utilization"], 0, 3).round(4)
        
        return df
    
    def create_match_features(self, df):
        """Create expertise and language match features"""
        # Expertise match
        def expertise_match(row):
            ins = row.get("insurance_type", "UNKNOWN")
            if ins == "auto":
                return int(row.get("expertise_auto", 0) == 1)
            elif ins == "home":
                return int(row.get("expertise_home", 0) == 1)
            elif ins == "bundle":
                return int(row.get("expertise_bundle", 0) == 1)
            return 0.5
        
        df["expertise_match"] = df.apply(expertise_match, axis=1)
        
        # Language match
        def language_match(row):
            lead_lang = row.get("language", "UNKNOWN")
            broker_lang = row.get("languages", "English")
            if broker_lang == "Bilingual":
                return 1.0
            if lead_lang == "UNKNOWN":
                return 0.8
            return 1.0 if lead_lang == broker_lang else 0.3
        
        df["language_match"] = df.apply(language_match, axis=1)
        
        return df
    
    def create_temporal_features(self, df):
        """Create temporal features from lead_date"""
        if "lead_date" in df.columns:
            lead_dt = pd.to_datetime(df["lead_date"])
            df["lead_dayofweek"] = lead_dt.dt.dayofweek
            df["lead_weekofyear"] = lead_dt.dt.isocalendar().week.fillna(0).astype(int)
            df["lead_quarter"] = lead_dt.dt.quarter
            df["lead_year"] = lead_dt.dt.year
        
        return df
    
    def create_broker_quality_score(self, df, fit=False):
        """Create normalized broker quality score"""
        quality_cols = ["conversion_rate", "csat_score", "skill_level", "reliability"]
        
        for col in quality_cols:
            if col in df.columns:
                if fit:
                    # Calculate min/max from training data
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:
                        df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)
                        # Store for test set
                        self.quality_scales[col] = {"min": min_val, "max": max_val}
                    else:
                        df[f"{col}_norm"] = 0.5
                        self.quality_scales[col] = {"min": min_val, "max": max_val}
                    logger.info(f"Fit scale for {col}: min={min_val:.3f}, max={max_val:.3f}")
                else:
                    # Use stored scales for test data
                    if col in self.quality_scales:
                        min_val = self.quality_scales[col]["min"]
                        max_val = self.quality_scales[col]["max"]
                        if max_val > min_val:
                            df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)
                        else:
                            df[f"{col}_norm"] = 0.5
        
        broker_quality_cols = [c for c in ["conversion_rate_norm", "csat_score_norm", "skill_level_norm", "reliability_norm"] 
                              if c in df.columns]
        
        if broker_quality_cols:
            df["broker_quality_score"] = df[broker_quality_cols].mean(axis=1).round(4)
        
        # Create quality x value feature
        if "broker_quality_score" in df.columns and "quote_value" in df.columns:
            if fit:
                self.max_quote_value = df["quote_value"].max()
                df["quality_x_value"] = (df["broker_quality_score"] * df["quote_value"] / self.max_quote_value).round(4)
                logger.info(f"Fit max_quote_value: {self.max_quote_value:.3f}")
            else:
                if self.max_quote_value is not None:
                    df["quality_x_value"] = (df["broker_quality_score"] * df["quote_value"] / self.max_quote_value).round(4)
        
        return df
    
    def create_claims_risk(self, df):
        """Create ordinal encoding for claims severity"""
        if "claims_severity" in df.columns:
            claims_risk_map = {"none": 0, "minor": 1, "major": 2}
            df["claims_risk"] = df["claims_severity"].map(claims_risk_map).fillna(0).astype(int)
        
        return df
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        if "ribo_licensed" in df.columns and "expertise_match" in df.columns:
            df["ribo_x_expertise"] = df["ribo_licensed"] * df["expertise_match"]
        
        if "claims_risk" in df.columns and "skill_level" in df.columns:
            df["claims_x_skill"] = df["claims_risk"] * df["skill_level"]
        
        if "tenure_years" in df.columns and "broker_quality_score" in df.columns:
            df["tenure_x_quality"] = df["tenure_years"] * df["broker_quality_score"]
        
        return df
    
    def create_region_mismatch(self, df):
        """Create region mismatch flag"""
        if "region_x" in df.columns and "region_y" in df.columns:
            df["region_mismatch"] = (df["region_x"] != df["region_y"]).astype(int)
        
        return df
    
    def apply_log_transforms(self, df):
        """Apply log1p transforms to skewed features"""
        log_transform_cols = [
            "conversion_delay_days",
            "response_time_hours",
            "quote_value",
            "patience_hours",
        ]
        
        for col in log_transform_cols:
            if col in df.columns:
                df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
        
        return df
    
    def cap_outliers(self, df, fit=False):
        """Cap outliers at 99th and 1st percentiles"""
        outlier_cap_cols = [
            "log_conversion_delay_days",
            "log_response_time_hours",
            "profit",
            "expected_profit",
            "log_patience_hours",
            "log_quote_value",
        ]
        
        for col in outlier_cap_cols:
            if col in df.columns:
                if fit:
                    # Calculate percentiles from training data
                    p99 = df[col].quantile(0.99)
                    p01 = df[col].quantile(0.01)
                    self.outlier_bounds[col] = {"p01": p01, "p99": p99}
                    df[col] = np.clip(df[col], p01, p99)
                    logger.info(f"Fit bounds for {col}: p01={p01:.3f}, p99={p99:.3f}")
                else:
                    # Use stored bounds for test data
                    if col in self.outlier_bounds:
                        p01 = self.outlier_bounds[col]["p01"]
                        p99 = self.outlier_bounds[col]["p99"]
                        df[col] = np.clip(df[col], p01, p99)
        
        return df
    
    def encode_categorical(self, df, fit=False):
        """Apply one-hot and label encoding"""
        # One-hot encode languages (no fit/transform needed as it's deterministic)
        if "languages" in df.columns:
            lang_dummies = pd.get_dummies(df["languages"], prefix="lang", dtype=int)
            df = pd.concat([df, lang_dummies], axis=1)
        
        # Label encode selected columns
        le_cols = ["insurance_type", "market_regime", "claims_severity"]
        for col in le_cols:
            if col in df.columns:
                if fit:
                    # Fit label encoder on training data
                    le = LabelEncoder()
                    df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    logger.info(f"Fit LabelEncoder for {col}: {len(le.classes_)} classes")
                else:
                    # Transform test data using fitted encoder
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Handle unseen labels in test set
                        df[f"{col}_enc"] = df[col].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
        
        return df
    
    def drop_unnecessary_columns(self, df):
        """Drop columns that are not needed for modeling"""
        DROP_COLS = [
            "original_lead_id_x", "original_lead_id_y", "postal_code_prefix",
            "action_probabilities", "conversion_value", "current_caseload",
            "assignment_date", "lead_date", "market_regime", "claims_severity",
            "conversion_rate_norm", "csat_score_norm", "skill_level_norm", "reliability_norm",
            "response_time_bucket", "quote_value_tier",
        ]
        
        DROP_COLS = [c for c in DROP_COLS if c in df.columns]
        if DROP_COLS:
            df = df.drop(columns=DROP_COLS)
        
        return df
    
    def handle_responded_column(self, df):
        """Fill NaN in responded column with 0"""
        if "responded" in df.columns:
            df["responded"] = df["responded"].fillna(0).astype(int)
        
        return df
    
    def save_transformers(self):
        """Save all fitted transformers to disk"""
        logger.info("Saving fitted transformers...")
        
        transformers_dir = self.config.transformed_dir / "transformers"
        create_directories([transformers_dir])
        
        # Save median values
        if self.median_values:
            joblib.dump(self.median_values, transformers_dir / "median_imputer.pkl")
            logger.info(f"Saved median_imputer.pkl with {len(self.median_values)} features")
        
        # Save quality scales
        if self.quality_scales:
            joblib.dump(self.quality_scales, transformers_dir / "quality_scaler.pkl")
            logger.info(f"Saved quality_scaler.pkl with {len(self.quality_scales)} features")
        
        # Save outlier bounds
        if self.outlier_bounds:
            joblib.dump(self.outlier_bounds, transformers_dir / "outlier_capper.pkl")
            logger.info(f"Saved outlier_capper.pkl with {len(self.outlier_bounds)} features")
        
        # Save label encoders
        if self.label_encoders:
            joblib.dump(self.label_encoders, transformers_dir / "label_encoders.pkl")
            logger.info(f"Saved label_encoders.pkl with {len(self.label_encoders)} encoders")
        
        # Save max quote value
        if self.max_quote_value is not None:
            joblib.dump(self.max_quote_value, transformers_dir / "max_quote_value.pkl")
            logger.info(f"Saved max_quote_value.pkl: {self.max_quote_value:.3f}")
        
        logger.info(f"All transformers saved to {transformers_dir}")
        return self
    
    def save_numpy_arrays(self):
        """Save train and test sets as numpy arrays for model training/evaluation."""
        logger.info("Saving numpy arrays for model training...")
        
        # Define feature lists (must match what the model expects)
        self.client_features = [
            "quote_value", "lead_difficulty", "sophistication", "patience_hours",
            "digital_engagement_score", "tenure_years", "log_quote_value", "log_patience_hours",
            "month", "hour_of_day", "lead_dayofweek", "lead_quarter", "is_weekend",
            "insurance_type_enc", "claims_risk", "multi_product_intent",
            "insurance_type_missing", "language_missing", "tenure_years_missing",
            "digital_engagement_score_missing"
        ]
        
        self.broker_features = [
            "skill_level", "conversion_rate", "csat_score", "reliability", "efficiency",
            "avg_response_time", "burnout_risk", "commission_rate", "cost_per_lead",
            "utilization", "ribo_licensed", "is_new_broker", "expertise_auto",
            "expertise_home", "expertise_bundle", "broker_quality_score",
            "lang_Bilingual", "lang_English", "lang_French"
        ]
        
        self.interaction_features = [
            "expertise_match", "language_match", "workload_ratio", "quality_x_value",
            "position_bias", "interaction_number", "responded", "response_time_bucket_ord",
            "log_response_time_hours", "ribo_x_expertise", "claims_x_skill", "tenure_x_quality"
        ]
        
        # Filter to columns that exist in the DataFrames
        self.client_features = [c for c in self.client_features if c in self.train_df.columns]
        self.broker_features = [c for c in self.broker_features if c in self.train_df.columns]
        self.interaction_features = [c for c in self.interaction_features if c in self.train_df.columns]
        
        # Train arrays
        train_client = self.train_df[self.client_features].values.astype(np.float32)
        train_broker = self.train_df[self.broker_features].values.astype(np.float32)
        train_interaction = self.train_df[self.interaction_features].values.astype(np.float32)
        train_labels = self.train_df["converted"].values.astype(np.float32)
        
        # Test arrays
        test_client = self.test_df[self.client_features].values.astype(np.float32)
        test_broker = self.test_df[self.broker_features].values.astype(np.float32)
        test_interaction = self.test_df[self.interaction_features].values.astype(np.float32)
        test_labels = self.test_df["converted"].values.astype(np.float32)
        
        # Save to numpy directory
        numpy_dir = self.config.transformed_dir
        np.save(numpy_dir / "train_client.npy", train_client)
        np.save(numpy_dir / "train_broker.npy", train_broker)
        np.save(numpy_dir / "train_interaction.npy", train_interaction)
        np.save(numpy_dir / "train_labels.npy", train_labels)
        
        np.save(numpy_dir / "test_client.npy", test_client)
        np.save(numpy_dir / "test_broker.npy", test_broker)
        np.save(numpy_dir / "test_interaction.npy", test_interaction)
        np.save(numpy_dir / "test_labels.npy", test_labels)
        
        # Also save feature lists as JSON for the API server
        feature_lists = {
            "client_features": self.client_features,
            "broker_features": self.broker_features,
            "interaction_features": self.interaction_features
        }
        with open(numpy_dir / "feature_lists.json", "w") as f:
            json.dump(feature_lists, f, indent=2)
        
        logger.info(f"Numpy arrays saved to {numpy_dir}")
        logger.info(f"Train shapes: client {train_client.shape}, broker {train_broker.shape}, inter {train_interaction.shape}")
        logger.info(f"Test shapes: client {test_client.shape}, broker {test_broker.shape}, inter {test_interaction.shape}")
        
        return self
    
    def transform_train_test(self):
        """Apply all transformations to train and test sets separately"""
        logger.info("Applying transformations to train and test sets...")
        
        # ========== TRANSFORM TRAIN SET (FIT) ==========
        logger.info("=" * 50)
        logger.info("Transforming TRAIN set (fitting transformations)...")
        logger.info("=" * 50)
        
        train_transformed = self.train_df.copy()
        
        # Apply transformations with fit=True
        train_transformed = self.create_missing_flags(train_transformed)
        train_transformed = self.impute_categorical(train_transformed)
        train_transformed = self.impute_numeric(train_transformed, fit=True)
        train_transformed = self.create_derived_features(train_transformed)
        train_transformed = self.create_response_time_buckets(train_transformed)
        train_transformed = self.create_workload_ratio(train_transformed)
        train_transformed = self.create_match_features(train_transformed)
        train_transformed = self.create_temporal_features(train_transformed)
        train_transformed = self.create_broker_quality_score(train_transformed, fit=True)
        train_transformed = self.create_claims_risk(train_transformed)
        train_transformed = self.create_interaction_features(train_transformed)
        train_transformed = self.create_region_mismatch(train_transformed)
        train_transformed = self.apply_log_transforms(train_transformed)
        train_transformed = self.cap_outliers(train_transformed, fit=True)
        train_transformed = self.encode_categorical(train_transformed, fit=True)
        train_transformed = self.drop_unnecessary_columns(train_transformed)
        train_transformed = self.handle_responded_column(train_transformed)
        
        self.train_df = train_transformed
        
        # ========== TRANSFORM TEST SET (TRANSFORM ONLY) ==========
        logger.info("=" * 50)
        logger.info("Transforming TEST set (applying fitted transformations)...")
        logger.info("=" * 50)
        
        test_transformed = self.test_df.copy()
        
        # Apply transformations with fit=False
        test_transformed = self.create_missing_flags(test_transformed)
        test_transformed = self.impute_categorical(test_transformed)
        test_transformed = self.impute_numeric(test_transformed, fit=False)
        test_transformed = self.create_derived_features(test_transformed)
        test_transformed = self.create_response_time_buckets(test_transformed)
        test_transformed = self.create_workload_ratio(test_transformed)
        test_transformed = self.create_match_features(test_transformed)
        test_transformed = self.create_temporal_features(test_transformed)
        test_transformed = self.create_broker_quality_score(test_transformed, fit=False)
        test_transformed = self.create_claims_risk(test_transformed)
        test_transformed = self.create_interaction_features(test_transformed)
        test_transformed = self.create_region_mismatch(test_transformed)
        test_transformed = self.apply_log_transforms(test_transformed)
        test_transformed = self.cap_outliers(test_transformed, fit=False)
        test_transformed = self.encode_categorical(test_transformed, fit=False)
        test_transformed = self.drop_unnecessary_columns(test_transformed)
        test_transformed = self.handle_responded_column(test_transformed)
        
        self.test_df = test_transformed
        
        logger.info("=" * 50)
        logger.info("Transformations complete!")
        logger.info(f"Train set shape: {self.train_df.shape}")
        logger.info(f"Test set shape: {self.test_df.shape}")
        logger.info("=" * 50)
        
        # Save all fitted transformers
        self.save_transformers()
        
        # Save numpy arrays for model training/evaluation
        self.save_numpy_arrays()
        
        return self
    
    def split_positive_negative(self):
        """Split into positive (assigned) and negative (unassigned) samples"""
        logger.info("Splitting positive and negative samples...")
        
        self.df_positive = self.df[self.df["is_assigned"] == 1].copy()
        self.df_negative = self.df[self.df["is_assigned"] == 0].copy()
        
        logger.info(f"Positive (assigned) samples: {len(self.df_positive):,}")
        logger.info(f"Negative (unassigned) samples: {len(self.df_negative):,}")
        
        if len(self.df_positive) > 0:
            conversion_rate = self.df_positive['converted'].mean() * 100
            logger.info(f"Conversion rate (positive): {conversion_rate:.2f}%")
        
        return self
    
    def save_transformed_data(self):
        """Save all transformed datasets as CSV (for inspection)"""
        logger.info("Saving transformed data as CSV...")
        
        # Create transformed directory
        create_directories([self.config.transformed_dir])
        
        # Create full transformed dataset by combining train and test
        full_transformed = pd.concat([self.train_df, self.test_df], axis=0, ignore_index=True)
        
        # Save datasets
        self.train_df.to_csv(self.config.transformed_dir / "train_v81.csv", index=False)
        self.test_df.to_csv(self.config.transformed_dir / "test_v81.csv", index=False)
        full_transformed.to_csv(self.config.transformed_dir / "prepared_full_v81.csv", index=False)
        
        # Save original data for reference
        self.df_positive.to_csv(self.config.transformed_dir / "prepared_positive_v81.csv", index=False)
        self.df_negative.to_csv(self.config.transformed_dir / "prepared_negative_v81.csv", index=False)
        self.leads_full.to_csv(self.config.transformed_dir / "leads_full_v81.csv", index=False)
        self.brokers_full.to_csv(self.config.transformed_dir / "brokers_full_v81.csv", index=False)
        self.counterfactual_clean.to_csv(self.config.transformed_dir / "counterfactual_clean_v81.csv", index=False)
        
        logger.info("✓ Data preparation complete. Ready for model training.")
        logger.info(f"  train_v81.csv                 — {len(self.train_df):,} rows (transformed, 70%)")
        logger.info(f"  test_v81.csv                  — {len(self.test_df):,} rows (transformed, 30%)")
        logger.info(f"  prepared_full_v81.csv         — {len(full_transformed):,} rows × {full_transformed.shape[1]} cols")
        logger.info(f"  prepared_positive_v81.csv     — {len(self.df_positive):,} rows (original)")
        logger.info(f"  prepared_negative_v81.csv     — {len(self.df_negative):,} rows (original)")
        logger.info(f"  leads_full_v81.csv            — {len(self.leads_full):,} rows")
        logger.info(f"  brokers_full_v81.csv          — {len(self.brokers_full):,} rows")
        logger.info(f"  counterfactual_clean_v81.csv  — {len(self.counterfactual_clean):,} rows")
        logger.info(f"  transformers/                 — Fitted transformers for inference")
        
        return self