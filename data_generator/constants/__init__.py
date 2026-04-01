from pathlib import Path

CONFIG_FILE_PATH = Path("configs/config.yaml")

# Output directories
ARTIFACT_DIR = Path("artifacts")
DATA_DIR = ARTIFACT_DIR / "data"
LOG_DIR = ARTIFACT_DIR / "logs"

# Ensure directories exist
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Column names
BROKER_COLUMNS = [
    'broker_id', 'region', 'expertise_auto', 'expertise_home',
    'expertise_bundle', 'conversion_rate', 'csat_score', 'languages',
    'ribo_licensed', 'ribo_license_years', 'capacity', 'avg_response_time',
    'is_new_broker', 'skill_level', 'reliability', 'commission_rate',
    'cost_per_lead', 'efficiency', 'burnout_risk'
]

# Output file names
BROKERS_FILE = "synthetic_brokers_v80.csv"
LEADS_FILE = "synthetic_leads_v80.csv"
ASSIGNMENTS_FILE = "synthetic_assignments_v80.csv"
COUNTERFACTUAL_FILE = "synthetic_counterfactual_v80.csv"
HISTORICAL_FILE = "synthetic_historical_v80.csv"

# Full paths
BROKERS_PATH = DATA_DIR / BROKERS_FILE
LEADS_PATH = DATA_DIR / LEADS_FILE
ASSIGNMENTS_PATH = DATA_DIR / ASSIGNMENTS_FILE
COUNTERFACTUAL_PATH = DATA_DIR / COUNTERFACTUAL_FILE
HISTORICAL_PATH = DATA_DIR / HISTORICAL_FILE

# File info dictionary
OUTPUT_FILES = {
    'brokers': {
        'path': BROKERS_PATH,
        'filename': BROKERS_FILE,
        'description': 'broker profiles'
    },
    'leads': {
        'path': LEADS_PATH,
        'filename': LEADS_FILE,
        'description': 'lead records'
    },
    'assignments': {
        'path': ASSIGNMENTS_PATH,
        'filename': ASSIGNMENTS_FILE,
        'description': 'full logged policy data'
    },
    'counterfactual': {
        'path': COUNTERFACTUAL_PATH,
        'filename': COUNTERFACTUAL_FILE,
        'description': 'evaluation only (ground truth)'
    },
    'historical': {
        'path': HISTORICAL_PATH,
        'filename': HISTORICAL_FILE,
        'description': 'merged training dataset'
    }
}

# Validation thresholds
CONVERSION_RATE_MIN = 5.0
CONVERSION_RATE_MAX = 25.0
PROPENSITY_SCORE_MIN = 0.0
PROPENSITY_SCORE_MAX = 1.0