import os
import sys
import logging
import datetime
from pathlib import Path
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"

log_dir = "logs"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

# Create timestamp for this run
timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
log_filepath = os.path.join(log_dir, f"running_logs_{timestamp}.log")

os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(
    level = logging.INFO,
    format = logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("agent-recommender")