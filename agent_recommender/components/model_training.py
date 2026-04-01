import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, f1_score
from pathlib import Path
import json
from agent_recommender import logger
from agent_recommender.entity.config_entity import ModelTrainingConfig
from agent_recommender.utils.utility import create_directories


class BrokerMatchDataset(Dataset):
    def __init__(self, df, client_cols, broker_cols, interaction_cols):
        # Ensure all features are numeric
        client_df = df[client_cols].copy()
        broker_df = df[broker_cols].copy()
        interaction_df = df[interaction_cols].copy()
        
        # Convert any object columns to float
        for col in client_df.columns:
            if client_df[col].dtype == 'object':
                client_df[col] = pd.to_numeric(client_df[col], errors='coerce').fillna(0)
            client_df[col] = client_df[col].astype(float)
        
        for col in broker_df.columns:
            if broker_df[col].dtype == 'object':
                broker_df[col] = pd.to_numeric(broker_df[col], errors='coerce').fillna(0)
            broker_df[col] = broker_df[col].astype(float)
        
        for col in interaction_df.columns:
            if interaction_df[col].dtype == 'object':
                interaction_df[col] = pd.to_numeric(interaction_df[col], errors='coerce').fillna(0)
            interaction_df[col] = interaction_df[col].astype(float)
        
        # Convert to tensors
        self.client_x = torch.tensor(client_df.values, dtype=torch.float32)
        self.broker_x = torch.tensor(broker_df.values, dtype=torch.float32)
        self.interaction_x = torch.tensor(interaction_df.values, dtype=torch.float32)
        self.labels = torch.tensor(df["converted"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.client_x[idx],
            self.broker_x[idx],
            self.interaction_x[idx],
            self.labels[idx],
        )


class Tower(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, embed_dim),
        )

    def forward(self, x):
        emb = self.net(x)
        return F.normalize(emb, p=2, dim=1)


class TwoTowerModel(nn.Module):
    def __init__(self, client_dim, broker_dim, interaction_dim, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.client_tower = Tower(client_dim, hidden_dim, embed_dim, dropout)
        self.broker_tower = Tower(broker_dim, hidden_dim, embed_dim, dropout)

        fusion_input_dim = embed_dim + embed_dim + embed_dim + 1 + interaction_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, client_x, broker_x, interaction_x):
        client_emb = self.client_tower(client_x)
        broker_emb = self.broker_tower(broker_x)
        hadamard = client_emb * broker_emb
        dot = hadamard.sum(dim=1, keepdim=True)
        fusion_input = torch.cat([client_emb, broker_emb, hadamard, dot, interaction_x], dim=1)
        return self.fusion_head(fusion_input).squeeze(1)


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.optimal_threshold = 0.5
        
        # Feature columns (from notebook)
        self.CLIENT_FEATURES = [
            "quote_value", "lead_difficulty", "sophistication", "patience_hours",
            "digital_engagement_score", "tenure_years",
            "log_quote_value", "log_patience_hours",
            "month", "hour_of_day", "lead_dayofweek", "lead_quarter",
            "is_weekend", "insurance_type_enc", "claims_risk",
            "multi_product_intent", "insurance_type_missing", "language_missing",
            "tenure_years_missing", "digital_engagement_score_missing",
        ]
        
        self.BROKER_FEATURES = [
            "skill_level", "conversion_rate", "csat_score", "reliability",
            "efficiency", "avg_response_time", "burnout_risk",
            "commission_rate", "cost_per_lead", "utilization",
            "ribo_licensed", "is_new_broker",
            "expertise_auto", "expertise_home", "expertise_bundle",
            "broker_quality_score",
            "lang_Bilingual", "lang_English", "lang_French",
        ]
        
        self.INTERACTION_FEATURES = [
            "expertise_match", "language_match", "workload_ratio", "quality_x_value",
            "position_bias", "interaction_number",
            "responded", "response_time_bucket_ord", "log_response_time_hours",
            "ribo_x_expertise", "claims_x_skill", "tenure_x_quality",
        ]
        
        # Set random seeds
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
    def load_data(self):
        """Load transformed data from data_transformation stage"""
        logger.info("Loading transformed data...")
        
        transformed_dir = self.config.transformed_data_dir
        
        # Check if the files exist
        train_path = transformed_dir / "train_v81.csv"
        test_path = transformed_dir / "test_v81.csv"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        # Load train and test sets
        self.train_df = pd.read_csv(train_path)
        test_df_full = pd.read_csv(test_path)
        
        logger.info(f"Loaded train set: {len(self.train_df):,} rows")
        logger.info(f"Loaded test set: {len(test_df_full):,} rows")
        
        # Split test into validation and test (50/50)
        # This gives us 70% train, 15% val, 15% test
        self.val_df, self.test_df = train_test_split(
            test_df_full, 
            test_size=0.5, 
            random_state=self.config.seed,
            stratify=test_df_full["converted"]
        )
        
        total_samples = len(self.train_df) + len(self.val_df) + len(self.test_df)
        logger.info(f"Train set: {len(self.train_df):,} rows ({len(self.train_df)/total_samples*100:.1f}%)")
        logger.info(f"Validation set: {len(self.val_df):,} rows ({len(self.val_df)/total_samples*100:.1f}%)")
        logger.info(f"Test set: {len(self.test_df):,} rows ({len(self.test_df)/total_samples*100:.1f}%)")
        
        # Verify conversion rates
        logger.info(f"Train conversion rate: {self.train_df['converted'].mean()*100:.2f}%")
        logger.info(f"Val conversion rate: {self.val_df['converted'].mean()*100:.2f}%")
        logger.info(f"Test conversion rate: {self.test_df['converted'].mean()*100:.2f}%")
        
        # Check data types
        logger.info("Checking data types...")
        for df_name, df in [("Train", self.train_df), ("Val", self.val_df), ("Test", self.test_df)]:
            object_cols = df.select_dtypes(include=['object']).columns.tolist()
            if object_cols:
                logger.warning(f"{df_name} has object columns: {object_cols[:5]}...")  # Show first 5
        
        return self
    
    def filter_columns(self):
        """Filter feature columns to only those that exist in data and ensure numeric types"""
        def filter_cols(cols, df):
            missing = [c for c in cols if c not in df.columns]
            if missing:
                logger.warning(f"Missing columns (skipped): {missing}")
            return [c for c in cols if c in df.columns]
        
        self.CLIENT_FEATURES = filter_cols(self.CLIENT_FEATURES, self.train_df)
        self.BROKER_FEATURES = filter_cols(self.BROKER_FEATURES, self.train_df)
        self.INTERACTION_FEATURES = filter_cols(self.INTERACTION_FEATURES, self.train_df)
        
        # Convert all feature columns to numeric
        for df in [self.train_df, self.val_df, self.test_df]:
            for cols in [self.CLIENT_FEATURES, self.BROKER_FEATURES, self.INTERACTION_FEATURES]:
                for col in cols:
                    if col in df.columns:
                        # Convert to numeric, coercing errors to NaN, then fill NaN with 0
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        # Ensure float32
                        df[col] = df[col].astype(np.float32)
        
        logger.info(f"Client features: {len(self.CLIENT_FEATURES)}")
        logger.info(f"Broker features: {len(self.BROKER_FEATURES)}")
        logger.info(f"Interaction features: {len(self.INTERACTION_FEATURES)}")
        
        return self
    
    def create_dataloaders(self):
        """Create PyTorch dataloaders"""
        logger.info("Creating dataloaders...")
        
        # Create datasets
        train_ds = BrokerMatchDataset(self.train_df, self.CLIENT_FEATURES, self.BROKER_FEATURES, self.INTERACTION_FEATURES)
        val_ds = BrokerMatchDataset(self.val_df, self.CLIENT_FEATURES, self.BROKER_FEATURES, self.INTERACTION_FEATURES)
        test_ds = BrokerMatchDataset(self.test_df, self.CLIENT_FEATURES, self.BROKER_FEATURES, self.INTERACTION_FEATURES)
        
        # WeightedRandomSampler for balanced batches
        labels_array = self.train_df["converted"].values.astype(int)
        class_counts = np.bincount(labels_array)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels_array]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(sample_weights),
            replacement=True,
        )
        
        self.train_loader = DataLoader(
            train_ds, 
            batch_size=self.config.batch_size, 
            sampler=sampler, 
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_ds, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        self.test_loader = DataLoader(
            test_ds, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info(f"Test batches: {len(self.test_loader)}")
        logger.info(f"Class counts — neg: {class_counts[0]:,}  pos: {class_counts[1]:,}  ratio: {class_counts[0]/class_counts[1]:.1f}x")
        
        return self
    
    def build_model(self):
        """Build the two-tower model"""
        logger.info("Building model...")
        logger.info(f"Device: {self.device}")
        
        self.model = TwoTowerModel(
            client_dim=len(self.CLIENT_FEATURES),
            broker_dim=len(self.BROKER_FEATURES),
            interaction_dim=len(self.INTERACTION_FEATURES),
            embed_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params:,}")
        
        return self
    
    def setup_training(self):
        """Setup loss function, optimizer, and scheduler"""
        logger.info("Setting up training...")
        
        self.loss_fn = FocalLoss(
            alpha=self.config.focal_alpha, 
            gamma=self.config.focal_gamma
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode="max", 
            factor=0.5, 
            patience=5
        )
        
        logger.info(f"Focal Loss: alpha={self.config.focal_alpha}, gamma={self.config.focal_gamma}")
        logger.info(f"Optimizer: AdamW lr={self.config.learning_rate} weight_decay=1e-4")
        
        return self
    
    def run_epoch(self, loader, train=True):
        """Run one epoch of training or validation"""
        self.model.train(train)
        total_loss, all_probs, all_labels = 0.0, [], []
        
        with torch.set_grad_enabled(train):
            for batch in loader:
                client_x, broker_x, inter_x, labels = batch
                client_x = client_x.to(self.device)
                broker_x = broker_x.to(self.device)
                inter_x = inter_x.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(client_x, broker_x, inter_x)
                loss = self.loss_fn(logits, labels)
                
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                total_loss += loss.item() * len(labels)
                all_probs.extend(torch.sigmoid(logits).cpu().detach().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(all_labels)
        auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
        
        return avg_loss, auc, pr_auc
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        best_val_auc = 0.0
        best_epoch = 0
        history = []
        
        # Create model directory
        create_directories([self.config.model_dir])
        
        logger.info(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train AUC':>10}  {'Val Loss':>9}  {'Val AUC':>8}  {'Val PR-AUC':>10}")
        logger.info("-" * 65)
        
        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_auc, train_pr = self.run_epoch(self.train_loader, train=True)
            val_loss, val_auc, val_pr = self.run_epoch(self.val_loader, train=False)
            
            self.scheduler.step(val_auc)
            
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auc": train_auc,
                "train_pr": train_pr,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_pr": val_pr,
            })
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                # Save best model
                torch.save(self.model.state_dict(), self.config.model_dir / "two_tower_best.pt")
            
            logger.info(
                f"{epoch:>5}  {train_loss:>10.4f}  {train_auc:>10.4f}  "
                f"{val_loss:>9.4f}  {val_auc:>8.4f}  {val_pr:>10.4f}"
                + ("  ← best" if epoch == best_epoch else "")
            )
        
        logger.info(f"\nBest model: epoch {best_epoch}  |  val AUC = {best_val_auc:.4f}")
        
        # Save training history
        create_directories([self.config.reports_dir])
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.config.reports_dir / "training_history.csv", index=False)
        
        return self
    
    def find_optimal_threshold(self):
        """Find optimal threshold on validation set"""
        logger.info("Finding optimal threshold...")
        
        # Load best model
        model_path = self.config.model_dir / "two_tower_best.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        val_probs, val_labels_list = [], []
        with torch.no_grad():
            for client_x, broker_x, inter_x, labels in self.val_loader:
                logits = self.model(
                    client_x.to(self.device), 
                    broker_x.to(self.device), 
                    inter_x.to(self.device)
                )
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels_list.extend(labels.numpy())
        
        val_probs_arr = np.array(val_probs)
        val_labels_arr = np.array(val_labels_list)
        
        thresholds = np.arange(0.05, 0.90, 0.01)
        best_t, best_f1 = 0.5, 0.0
        
        for t in thresholds:
            preds = (val_probs_arr >= t).astype(int)
            f1 = f1_score(val_labels_arr, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        
        self.optimal_threshold = best_t
        logger.info(f"Optimal threshold: {best_t:.2f} (val F1 = {best_f1:.3f})")
        
        return self
    
    def evaluate(self):
        """Evaluate model on test set"""
        logger.info("Evaluating on test set...")
        
        # Load best model
        model_path = self.config.model_dir / "two_tower_best.pt"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Get predictions
        test_loss, test_auc, test_pr_auc = self.run_epoch(self.test_loader, train=False)
        
        all_probs, all_labels_test = [], []
        with torch.no_grad():
            for client_x, broker_x, inter_x, labels in self.test_loader:
                logits = self.model(
                    client_x.to(self.device), 
                    broker_x.to(self.device), 
                    inter_x.to(self.device)
                )
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels_test.extend(labels.numpy())
        
        all_probs_arr = np.array(all_probs)
        all_labels_arr = np.array(all_labels_test)
        
        # Predictions with tuned threshold
        all_preds_tuned = (all_probs_arr >= self.optimal_threshold).astype(int)
        all_preds_half = (all_probs_arr >= 0.50).astype(int)
        
        # Generate classification reports
        report_tuned = classification_report(all_labels_arr, all_preds_tuned, target_names=["no convert", "convert"], output_dict=True)
        report_half = classification_report(all_labels_arr, all_preds_half, target_names=["no convert", "convert"], output_dict=True)
        
        # Save results
        results = {
            "test_loss": test_loss,
            "test_auc": test_auc,
            "test_pr_auc": test_pr_auc,
            "optimal_threshold": float(self.optimal_threshold),
            "tuned_threshold_report": report_tuned,
            "default_threshold_report": report_half,
            "probability_stats": {
                "mean_prob_positives": float(all_probs_arr[all_labels_arr == 1].mean()),
                "mean_prob_negatives": float(all_probs_arr[all_labels_arr == 0].mean())
            }
        }
        
        # Save results to JSON
        with open(self.config.reports_dir / "test_results.json", "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info("=" * 60)
        logger.info("TEST SET RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Loss:              {test_loss:.4f}")
        logger.info(f"  AUC-ROC:           {test_auc:.4f}")
        logger.info(f"  PR-AUC:            {test_pr_auc:.4f}")
        logger.info(f"  Threshold used:    {self.optimal_threshold:.2f} (tuned on val)")
        logger.info(f"  Mean prob positives: {results['probability_stats']['mean_prob_positives']:.3f}")
        logger.info(f"  Mean prob negatives: {results['probability_stats']['mean_prob_negatives']:.3f}")
        
        # Print classification report summary
        logger.info("\n--- With tuned threshold ---")
        logger.info(f"Precision (convert): {report_tuned['convert']['precision']:.3f}")
        logger.info(f"Recall (convert): {report_tuned['convert']['recall']:.3f}")
        logger.info(f"F1-score (convert): {report_tuned['convert']['f1-score']:.3f}")
        
        return self
    
    def save_model(self):
        """Save the final model and configuration"""
        logger.info("Saving model...")
        
        # Save model architecture config
        model_config = {
            "client_dim": len(self.CLIENT_FEATURES),
            "broker_dim": len(self.BROKER_FEATURES),
            "interaction_dim": len(self.INTERACTION_FEATURES),
            "embedding_dim": self.config.embedding_dim,
            "hidden_dim": self.config.hidden_dim,
            "dropout": self.config.dropout,
            "optimal_threshold": float(self.optimal_threshold),
            "client_features": self.CLIENT_FEATURES,
            "broker_features": self.BROKER_FEATURES,
            "interaction_features": self.INTERACTION_FEATURES,
            "focal_alpha": self.config.focal_alpha,
            "focal_gamma": self.config.focal_gamma,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "seed": self.config.seed
        }
        
        with open(self.config.model_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=4)
        
        logger.info(f"Model saved to {self.config.model_dir}")
        logger.info(f"  - two_tower_best.pt (model weights)")
        logger.info(f"  - model_config.json (model configuration)")
        
        return self