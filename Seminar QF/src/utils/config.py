from pathlib import Path

# Get the project root directory (assuming this file is in src/utils/config.py)
# src/utils -> src -> Seminar QF (project root)
# So parents should be:
# .parent -> utils
# .parent.parent -> src
# .parent.parent.parent -> Seminar QF

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
INTERMEDIATES_DIR = DATA_DIR / "intermediates"
DIAGNOSTICS_DIR = DATA_DIR / "diagnostics"

# Input File names
EQUITY_DATA_FILE = INPUT_DIR / "Jan2025_Accenture_Dataset_ErasmusCase.xlsx"
INTEREST_RATES_FILE = INPUT_DIR / "ECB Data Portal_20260125170805.csv"

# Ensure directories exist
INTERMEDIATES_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
