import json
import shutil
import re
from pathlib import Path
from datetime import datetime
from agent_recommender import logger
from agent_recommender.entity.config_entity import ModelPushConfig

class ModelPush:
    def __init__(self, config: ModelPushConfig):
        self.config = config
        self.new_metrics = self._load_metrics(config.model_dir / "metrics.json")
        self.best_metrics = self._load_best_metrics()
        self.version = None

    def _load_metrics(self, path):
        if path and path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def _load_best_metrics(self):
        best_dir = self.config.push_dir / "best"
        if best_dir.exists():
            metrics_file = best_dir / "metrics.json"
            return self._load_metrics(metrics_file)
        return None

    def _get_next_version(self):
        """Auto-increment patch version based on existing version directories."""
        existing = [d.name for d in self.config.push_dir.iterdir()
                    if d.is_dir() and re.match(r'v\d+\.\d+\.\d+', d.name)]
        if not existing:
            return "v1.0.0"
        # Sort by version tuple
        def version_key(v):
            return [int(x) for x in v[1:].split('.')]
        existing.sort(key=version_key)
        latest = existing[-1]
        parts = latest[1:].split('.')
        parts[-1] = str(int(parts[-1]) + 1)
        return f"v{'.'.join(parts)}"

    def is_new_model_better(self):
        if self.best_metrics is None:
            logger.info("No existing best model found. This model will become the champion.")
            return True
        # Compare primary metric: AUC (you can change to F1 or custom)
        new_auc = self.new_metrics.get("auc", 0)
        best_auc = self.best_metrics.get("auc", 0)
        return new_auc > best_auc

    def push_model(self):
        if not self.is_new_model_better():
            logger.info("New model is not better than current best. Skipping push.")
            return self

        self.version = self._get_next_version()
        logger.info(f"New model is better! Pushing as version {self.version}")

        push_dir = self.config.push_dir / self.version
        push_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        for file in ["two_tower_best.pt", "model_config.json", "metrics.json"]:
            src = self.config.model_dir / file
            if src.exists():
                shutil.copy2(src, push_dir / file)
            else:
                logger.warning(f"File {file} not found, skipping")

        # Create metadata
        metadata = {
            "version": self.version,
            "push_date": datetime.now().isoformat(),
            "metrics": self.new_metrics,
            "source_dir": str(self.config.model_dir)
        }
        with open(push_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Update best symlink
        best_link = self.config.push_dir / "best"
        if best_link.exists() or best_link.is_symlink():
            best_link.unlink()
        best_link.symlink_to(push_dir, target_is_directory=True)

        # Update latest symlink (optional, points to newest version)
        latest_link = self.config.push_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(push_dir, target_is_directory=True)

        # Update versions.json
        self._update_versions_json()

        logger.info(f"Model version {self.version} pushed and set as best.")
        return self

    def _update_versions_json(self):
        versions_file = self.config.push_dir / "versions.json"
        if versions_file.exists():
            with open(versions_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"versions": []}
        data["versions"].append({
            "version": self.version,
            "push_date": datetime.now().isoformat(),
            "path": str(self.config.push_dir / self.version),
            "metrics": self.new_metrics
        })
        with open(versions_file, 'w') as f:
            json.dump(data, f, indent=4)