import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from agent_recommender import logger
from agent_recommender.entity.config_entity import ModelEvaluationConfig
from agent_recommender.utils.utility import create_directories
from agent_recommender.components.model_training import TwoTowerModel, BrokerMatchDataset
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_config = None
        self.test_df = None
        
    def load_model(self):
        """Load trained model and configuration"""
        logger.info("Loading trained model...")
        
        # Load model configuration
        config_path = self.config.model_dir / "model_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.model_config = json.load(f)
        
        # Initialize model
        self.model = TwoTowerModel(
            client_dim=self.model_config['client_dim'],
            broker_dim=self.model_config['broker_dim'],
            interaction_dim=self.model_config['interaction_dim'],
            embed_dim=self.model_config['embedding_dim'],
            hidden_dim=self.model_config['hidden_dim'],
            dropout=self.model_config['dropout']
        ).to(self.device)
        
        # Load weights
        model_path = self.config.model_dir / "two_tower_best.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Model config: {self.model_config}")
        
        return self
    
    def load_test_data(self):
        """Load test data"""
        logger.info("Loading test data...")
        
        test_path = self.config.transformed_data_dir / "test_v81.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        self.test_df = pd.read_csv(test_path)
        
        # Get feature columns from model config
        self.client_features = self.model_config['client_features']
        self.broker_features = self.model_config['broker_features']
        self.interaction_features = self.model_config['interaction_features']
        
        # Ensure all features are numeric
        for cols in [self.client_features, self.broker_features, self.interaction_features]:
            for col in cols:
                if col in self.test_df.columns:
                    self.test_df[col] = pd.to_numeric(self.test_df[col], errors='coerce').fillna(0)
                    self.test_df[col] = self.test_df[col].astype(np.float32)
        
        logger.info(f"Test data loaded: {len(self.test_df):,} rows")
        logger.info(f"Test conversion rate: {self.test_df['converted'].mean()*100:.2f}%")
        
        return self
    
    def generate_predictions(self):
        """Generate predictions on test data"""
        logger.info("Generating predictions...")
        
        # Create dataset and dataloader
        test_dataset = BrokerMatchDataset(
            self.test_df, 
            self.client_features, 
            self.broker_features, 
            self.interaction_features
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 512,
            shuffle=False
        )
        
        # Generate predictions
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for client_x, broker_x, inter_x, labels in test_loader:
                client_x = client_x.to(self.device)
                broker_x = broker_x.to(self.device)
                inter_x = inter_x.to(self.device)
                
                logits = self.model(client_x, broker_x, inter_x)
                probs = torch.sigmoid(logits)
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        self.predictions = np.array(all_probs)
        self.true_labels = np.array(all_labels)
        
        logger.info(f"Generated {len(self.predictions)} predictions")
        
        return self
    
    def calculate_metrics(self):
        """Calculate all evaluation metrics"""
        logger.info("Calculating metrics...")
        
        # Use threshold from config or from model config
        threshold = self.config.threshold if hasattr(self.config, 'threshold') else self.model_config.get('optimal_threshold', 0.5)
        
        # Generate binary predictions
        self.pred_labels = (self.predictions >= threshold).astype(int)
        
        # Calculate metrics
        self.metrics = {
            "threshold_used": float(threshold),
            "accuracy": float(accuracy_score(self.true_labels, self.pred_labels)),
            "precision": float(precision_score(self.true_labels, self.pred_labels, zero_division=0)),
            "recall": float(recall_score(self.true_labels, self.pred_labels, zero_division=0)),
            "f1_score": float(f1_score(self.true_labels, self.pred_labels, zero_division=0)),
            "auc_roc": float(roc_auc_score(self.true_labels, self.predictions)),
            "auc_pr": float(average_precision_score(self.true_labels, self.predictions)),
            "confusion_matrix": confusion_matrix(self.true_labels, self.pred_labels).tolist()
        }
        
        # Calculate additional metrics
        tn, fp, fn, tp = confusion_matrix(self.true_labels, self.pred_labels).ravel()
        self.metrics["true_negatives"] = int(tn)
        self.metrics["false_positives"] = int(fp)
        self.metrics["false_negatives"] = int(fn)
        self.metrics["true_positives"] = int(tp)
        self.metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
        self.metrics["negative_predictive_value"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0
        
        logger.info(f"Metrics calculated: AUC-ROC={self.metrics['auc_roc']:.4f}, F1={self.metrics['f1_score']:.4f}")
        
        return self
    
    def plot_confusion_matrix(self):
        """Plot and save confusion matrix"""
        logger.info("Plotting confusion matrix...")
        
        # Create confusion matrix plot
        cm = np.array(self.metrics["confusion_matrix"])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Convert', 'Convert'],
                    yticklabels=['No Convert', 'Convert'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        create_directories([self.config.reports_dir])
        plt.savefig(self.config.reports_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {self.config.reports_dir / 'confusion_matrix.png'}")
        
        return self
    
    def plot_roc_curve(self):
        """Plot and save ROC curve"""
        logger.info("Plotting ROC curve...")
        
        
        fpr, tpr, thresholds = roc_curve(self.true_labels, self.predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {self.metrics["auc_roc"]:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(self.config.reports_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {self.config.reports_dir / 'roc_curve.png'}")
        
        return self
    
    def plot_pr_curve(self):
        """Plot and save Precision-Recall curve"""
        logger.info("Plotting Precision-Recall curve...")
    
        
        precision, recall, thresholds = precision_recall_curve(self.true_labels, self.predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {self.metrics["auc_pr"]:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(self.config.reports_dir / "pr_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"PR curve saved to {self.config.reports_dir / 'pr_curve.png'}")
        
        return self
    
    def plot_prediction_distribution(self):
        """Plot prediction probability distribution"""
        logger.info("Plotting prediction distribution...")
        
        plt.figure(figsize=(10, 6))
        
        # Separate predictions by true label
        pos_probs = self.predictions[self.true_labels == 1]
        neg_probs = self.predictions[self.true_labels == 0]
        
        plt.hist(neg_probs, bins=50, alpha=0.5, label=f'No Convert (n={len(neg_probs)})', color='blue')
        plt.hist(pos_probs, bins=50, alpha=0.5, label=f'Convert (n={len(pos_probs)})', color='red')
        plt.axvline(x=self.metrics["threshold_used"], color='green', linestyle='--', 
                   label=f'Threshold = {self.metrics["threshold_used"]:.2f}')
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(self.config.reports_dir / "prediction_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Prediction distribution saved to {self.config.reports_dir / 'prediction_distribution.png'}")
        
        return self
    
    def generate_classification_report(self):
        """Generate detailed classification report"""
        logger.info("Generating classification report...")
        
        self.classification_report = classification_report(
            self.true_labels, 
            self.pred_labels,
            target_names=["No Convert", "Convert"],
            output_dict=True
        )
        
        # Save as JSON
        with open(self.config.reports_dir / "classification_report.json", "w") as f:
            json.dump(self.classification_report, f, indent=4)
        
        logger.info("Classification report saved")
        
        return self
    
    def save_metrics(self):
        """Save all metrics to file"""
        logger.info("Saving metrics...")
        
        # Add additional metadata
        full_metrics = {
            "model_info": {
                "model_type": "TwoTowerModel",
                "embedding_dim": self.model_config['embedding_dim'],
                "hidden_dim": self.model_config['hidden_dim'],
                "dropout": self.model_config['dropout']
            },
            "test_data_info": {
                "total_samples": len(self.test_df),
                "positive_samples": int(self.true_labels.sum()),
                "negative_samples": int(len(self.test_df) - self.true_labels.sum()),
                "positive_ratio": float(self.true_labels.mean())
            },
            "metrics": self.metrics,
            "classification_report": self.classification_report
        }
        
        # Save to JSON
        with open(self.config.metrics_file, "w") as f:
            json.dump(full_metrics, f, indent=4)
        
        # Also save as human-readable text
        with open(self.config.reports_dir / "metrics_summary.txt", "w") as f:
            f.write("=" * 60 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MODEL INFORMATION:\n")
            f.write(f"  Model Type: TwoTowerModel\n")
            f.write(f"  Embedding Dimension: {self.model_config['embedding_dim']}\n")
            f.write(f"  Hidden Dimension: {self.model_config['hidden_dim']}\n")
            f.write(f"  Dropout: {self.model_config['dropout']}\n\n")
            
            f.write("TEST DATA INFORMATION:\n")
            f.write(f"  Total Samples: {full_metrics['test_data_info']['total_samples']:,}\n")
            f.write(f"  Positive Samples: {full_metrics['test_data_info']['positive_samples']:,}\n")
            f.write(f"  Negative Samples: {full_metrics['test_data_info']['negative_samples']:,}\n")
            f.write(f"  Positive Ratio: {full_metrics['test_data_info']['positive_ratio']*100:.2f}%\n\n")
            
            f.write("KEY METRICS:\n")
            f.write(f"  Threshold Used: {self.metrics['threshold_used']:.2f}\n")
            f.write(f"  Accuracy: {self.metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {self.metrics['precision']:.4f}\n")
            f.write(f"  Recall: {self.metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {self.metrics['f1_score']:.4f}\n")
            f.write(f"  AUC-ROC: {self.metrics['auc_roc']:.4f}\n")
            f.write(f"  AUC-PR: {self.metrics['auc_pr']:.4f}\n")
            f.write(f"  Specificity: {self.metrics['specificity']:.4f}\n\n")
            
            f.write("CONFUSION MATRIX:\n")
            f.write(f"  True Negatives: {self.metrics['true_negatives']:,}\n")
            f.write(f"  False Positives: {self.metrics['false_positives']:,}\n")
            f.write(f"  False Negatives: {self.metrics['false_negatives']:,}\n")
            f.write(f"  True Positives: {self.metrics['true_positives']:,}\n\n")
            
            f.write("CLASSIFICATION REPORT:\n")
            f.write(f"  No Convert - Precision: {self.classification_report['No Convert']['precision']:.4f}\n")
            f.write(f"  No Convert - Recall: {self.classification_report['No Convert']['recall']:.4f}\n")
            f.write(f"  No Convert - F1: {self.classification_report['No Convert']['f1-score']:.4f}\n")
            f.write(f"  Convert - Precision: {self.classification_report['Convert']['precision']:.4f}\n")
            f.write(f"  Convert - Recall: {self.classification_report['Convert']['recall']:.4f}\n")
            f.write(f"  Convert - F1: {self.classification_report['Convert']['f1-score']:.4f}\n")
        
        logger.info(f"Metrics saved to {self.config.metrics_file}")
        
        return self