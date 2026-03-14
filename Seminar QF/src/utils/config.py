from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
CALIBRATION_DIR = OUTPUT_DIR / "calibration"
CDS_FILTER_DIR = DATA_DIR / "cds_filters"

EQUITY_DATA_FILE = INPUT_DIR / "Jan2025_Accenture_Dataset_ErasmusCase.xlsx"
INTEREST_RATES_FILE = INPUT_DIR / "ECB Data Portal_20260125170805.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
