import shutil
import json
import torch
from pathlib import Path
from datetime import datetime
from agent_recommender import logger
from agent_recommender.entity.config_entity import ModelPushConfig
from agent_recommender.utils.utility import create_directories


class ModelPush:
    def __init__(self, config: ModelPushConfig):
        self.config = config
        self.version = config.version
        
    def push_model(self):
        """Push model to production directory"""
        logger.info(f"Pushing model version {self.version} to production...")
        
        # Create push directory with version
        push_dir = self.config.push_dir / self.version
        create_directories([push_dir])
        
        # Copy model files
        model_files = ["two_tower_best.pt", "model_config.json"]
        
        for file in model_files:
            src = self.config.model_dir / file
            dst = push_dir / file
            
            if src.exists():
                shutil.copy2(src, dst)
                logger.info(f"Copied {file} to {dst}")
            else:
                logger.warning(f"File not found: {src}")
        
        # Create version metadata
        metadata = {
            "version": self.version,
            "push_date": datetime.now().isoformat(),
            "model_type": "TwoTowerModel",
            "source_dir": str(self.config.model_dir),
            "files": model_files
        }
        
        # Load model config if exists
        config_path = push_dir / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            metadata["model_config"] = model_config
        
        # Save metadata
        with open(push_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Metadata saved to {push_dir / 'metadata.json'}")
        
        return self
    
    def create_latest_symlink(self):
        """Create a 'latest' symlink pointing to the current version"""
        logger.info("Creating latest symlink...")
        
        latest_link = self.config.push_dir / "latest"
        current_version_dir = self.config.push_dir / self.version
        
        # Remove existing symlink if it exists
        if latest_link.exists():
            if latest_link.is_symlink():
                latest_link.unlink()
            elif latest_link.is_dir():
                shutil.rmtree(latest_link)
        
        # Create new symlink
        try:
            latest_link.symlink_to(current_version_dir, target_is_directory=True)
            logger.info(f"Created symlink {latest_link} -> {current_version_dir}")
        except Exception as e:
            logger.warning(f"Could not create symlink: {e}")
            logger.info("Continuing without symlink...")
        
        return self
    
    def save_version_info(self):
        """Save version information to a file"""
        logger.info("Saving version information...")
        
        version_file = self.config.push_dir / "versions.json"
        
        # Load existing versions if any
        if version_file.exists():
            with open(version_file, 'r') as f:
                versions = json.load(f)
        else:
            versions = {"versions": []}
        
        # Add new version
        versions["versions"].append({
            "version": self.version,
            "push_date": datetime.now().isoformat(),
            "path": str(self.config.push_dir / self.version)
        })
        
        # Keep only last 5 versions
        versions["versions"] = versions["versions"][-5:]
        
        # Save
        with open(version_file, 'w') as f:
            json.dump(versions, f, indent=4)
        
        logger.info(f"Version info saved to {version_file}")
        
        return self
    
    def validate_push(self):
        """Validate that model was pushed successfully"""
        logger.info("Validating model push...")
        
        push_dir = self.config.push_dir / self.version
        
        # Check if all required files exist
        required_files = ["two_tower_best.pt", "model_config.json", "metadata.json"]
        missing_files = []
        
        for file in required_files:
            if not (push_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Missing files in push: {missing_files}")
            raise FileNotFoundError(f"Push validation failed: missing {missing_files}")
        
        logger.info(f"Push validation successful! Model version {self.version} is ready for production.")
        
        return self