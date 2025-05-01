"""Config file for module."""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

import os

# Paths - respect environment variables if set
PROJ_ROOT = os.environ.get("PROJ_ROOT")
if PROJ_ROOT is None:
    PROJ_ROOT = Path(__file__).resolve().parents[1]
else:
    PROJ_ROOT = Path(PROJ_ROOT)
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Dataset configuration
DATASET = "uciml/default-of-credit-card-clients-dataset"

# Define directories - respect environment variables if set
DATA_DIR = os.environ.get("DATA_DIR")
if DATA_DIR is None:
    DATA_DIR = PROJ_ROOT / "data"
else:
    DATA_DIR = Path(DATA_DIR)
logger.info(f"DATA_DIR path is: {DATA_DIR}")

RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", DATA_DIR / "raw")
if not isinstance(RAW_DATA_DIR, Path):
    RAW_DATA_DIR = Path(RAW_DATA_DIR)

INTERIM_DATA_DIR = os.environ.get("INTERIM_DATA_DIR", DATA_DIR / "interim")
if not isinstance(INTERIM_DATA_DIR, Path):
    INTERIM_DATA_DIR = Path(INTERIM_DATA_DIR)

PROCESSED_DATA_DIR = os.environ.get("PROCESSED_DATA_DIR", DATA_DIR / "processed")
if not isinstance(PROCESSED_DATA_DIR, Path):
    PROCESSED_DATA_DIR = Path(PROCESSED_DATA_DIR)

# Models directory
MODELS_DIR = os.environ.get("MODELS_DIR", PROJ_ROOT / "models")
if not isinstance(MODELS_DIR, Path):
    MODELS_DIR = Path(MODELS_DIR)

# Reports directory
REPORTS_DIR = os.environ.get("REPORTS_DIR", PROJ_ROOT / "reports")
if not isinstance(REPORTS_DIR, Path):
    REPORTS_DIR = Path(REPORTS_DIR)

FIGURES_DIR = os.environ.get("FIGURES_DIR", REPORTS_DIR / "figures")
if not isinstance(FIGURES_DIR, Path):
    FIGURES_DIR = Path(FIGURES_DIR)

# Model name
MODEL_NAME = os.environ.get("MODEL_NAME", "credit-default-classifier")
logger.debug(f"MODEL_NAME: {MODEL_NAME}")

# Log all directory paths for debugging
logger.debug(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
logger.debug(f"INTERIM_DATA_DIR: {INTERIM_DATA_DIR}")
logger.debug(f"PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")
logger.debug(f"MODELS_DIR: {MODELS_DIR}")
logger.debug(f"REPORTS_DIR: {REPORTS_DIR}")
logger.debug(f"FIGURES_DIR: {FIGURES_DIR}")
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_NAME = "credit-default-classifier"

categorical = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]

target = "default.payment.next.month"
