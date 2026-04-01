from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd


@dataclass
class DataArtifact:
    """Data artifact entity for storing generated data paths"""
    brokers_path: Union[str, Path]
    leads_path: Union[str, Path]
    assignments_path: Union[str, Path]
    counterfactual_path: Union[str, Path]
    historical_path: Union[str, Path]
    
    def __post_init__(self):
        """Convert string paths to Path objects"""
        self.brokers_path = Path(self.brokers_path)
        self.leads_path = Path(self.leads_path)
        self.assignments_path = Path(self.assignments_path)
        self.counterfactual_path = Path(self.counterfactual_path)
        self.historical_path = Path(self.historical_path)
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary with string paths"""
        return {
            'brokers_path': str(self.brokers_path),
            'leads_path': str(self.leads_path),
            'assignments_path': str(self.assignments_path),
            'counterfactual_path': str(self.counterfactual_path),
            'historical_path': str(self.historical_path)
        }
    
    def exists(self) -> bool:
        """Check if all artifact files exist"""
        return all([
            self.brokers_path.exists(),
            self.leads_path.exists(),
            self.assignments_path.exists(),
            self.counterfactual_path.exists(),
            self.historical_path.exists()
        ])
    
    def get_missing_files(self) -> list:
        """Get list of missing artifact files"""
        missing = []
        if not self.brokers_path.exists():
            missing.append('brokers_path')
        if not self.leads_path.exists():
            missing.append('leads_path')
        if not self.assignments_path.exists():
            missing.append('assignments_path')
        if not self.counterfactual_path.exists():
            missing.append('counterfactual_path')
        if not self.historical_path.exists():
            missing.append('historical_path')
        return missing
    
    def get_file_sizes(self) -> Dict[str, int]:
        """Get file sizes for all artifacts in bytes"""
        sizes = {}
        for name, path in self.to_dict().items():
            path_obj = Path(path)
            if path_obj.exists():
                sizes[name] = path_obj.stat().st_size
            else:
                sizes[name] = 0
        return sizes
    
    def load_dataframe(self, artifact_name: str) -> Optional[pd.DataFrame]:
        """Load a specific artifact as a DataFrame"""
        path_map = {
            'brokers': self.brokers_path,
            'leads': self.leads_path,
            'assignments': self.assignments_path,
            'counterfactual': self.counterfactual_path,
            'historical': self.historical_path
        }
        
        path = path_map.get(artifact_name)
        if path and path.exists():
            try:
                return pd.read_csv(path)
            except Exception as e:
                print(f"Error loading {artifact_name}: {e}")
                return None
        return None
    
    def load_all_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Load all artifacts as DataFrames"""
        dataframes = {}
        for name in ['brokers', 'leads', 'assignments', 'counterfactual', 'historical']:
            df = self.load_dataframe(name)
            if df is not None:
                dataframes[name] = df
        return dataframes