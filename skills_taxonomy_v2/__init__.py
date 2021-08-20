"""skills_taxonomy_v2."""
import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


def get_yaml_config(file_path: Path) -> Optional[dict]:
    """Fetch yaml config and return as dict if it exists."""
    if file_path.exists():
        with open(file_path, "rt") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)


# Define project base directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# S3 bucket name for this project
BUCKET_NAME = "skills-taxonomy-v2"

# Define log output locations
info_out = str(PROJECT_DIR / "info.log")
error_out = str(PROJECT_DIR / "errors.log")

# Read log config file
_log_config_path = Path(__file__).parent.resolve() / "config/logging.yaml"
_logging_config = get_yaml_config(_log_config_path)
if _logging_config:
    logging.config.dictConfig(_logging_config)

# Define module logger
logger = logging.getLogger(__name__)

# base/global config
_base_config_path = Path(__file__).parent.resolve() / "config/base.yaml"
config = get_yaml_config(_base_config_path)

# BUCKET and METAFLOW_PROFILE
load_dotenv(f"{PROJECT_DIR}/.env.shared")

# Get file name of custom stop words for skills extraction
custom_stopwords_dir = f"{PROJECT_DIR}/skills_taxonomy_v2/utils/custom_stop_words.txt"
