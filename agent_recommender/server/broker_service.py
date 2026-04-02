import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from agent_recommender import logger

class BrokerService:
    def __init__(self, brokers_path: Path = Path("artifacts/data_transformation/transformed/brokers_full_v81.csv")):
        self.brokers_path = brokers_path
        self.brokers_df = None
        self._load_brokers()

    def _load_brokers(self):
        if self.brokers_path.exists():
            self.brokers_df = pd.read_csv(self.brokers_path)
            logger.info(f"Loaded {len(self.brokers_df)} brokers from {self.brokers_path}")
        else:
            logger.error(f"Brokers file not found: {self.brokers_path}")
            self.brokers_df = pd.DataFrame()

    def get_broker(self, broker_id: str) -> Optional[Dict]:
        """Return broker features as a dict."""
        if self.brokers_df is None or self.brokers_df.empty:
            return None
        row = self.brokers_df[self.brokers_df["broker_id"] == broker_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def list_brokers(self, limit: int = 100) -> List[Dict]:
        """Return a list of simplified broker info for frontend dropdown."""
        if self.brokers_df is None or self.brokers_df.empty:
            return []
        # Select relevant columns for display
        display_cols = ["broker_id"]
        # Add name if available, else use broker_id
        if "broker_name" in self.brokers_df.columns:
            display_cols.append("broker_name")
        if "expertise_auto" in self.brokers_df.columns:
            # Build expertise list
            pass  # We'll compute on the fly
        brokers_list = []
        for _, row in self.brokers_df.head(limit).iterrows():
            broker = {
                "broker_id": row["broker_id"],
                "name": row.get("broker_name", row["broker_id"])
            }
            # Build expertise list
            expertise = []
            if row.get("expertise_auto", 0) == 1:
                expertise.append("Auto")
            if row.get("expertise_home", 0) == 1:
                expertise.append("Home")
            if row.get("expertise_bundle", 0) == 1:
                expertise.append("Bundle")
            broker["expertise"] = expertise
            # Languages (from one-hot or original)
            if "languages" in row:
                broker["languages"] = row["languages"].split(",") if isinstance(row["languages"], str) else [row["languages"]]
            else:
                langs = []
                if row.get("lang_English", 0) == 1:
                    langs.append("English")
                if row.get("lang_French", 0) == 1:
                    langs.append("French")
                if row.get("lang_Bilingual", 0) == 1:
                    langs.append("Bilingual")
                broker["languages"] = langs if langs else ["English"]
            brokers_list.append(broker)
        return brokers_list