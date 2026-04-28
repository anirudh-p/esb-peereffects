from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "1_Data" / "Raw"
CLEANED_DIR = PROJECT_ROOT / "1_Data" / "Cleaned"
OUTPUT_DIR = PROJECT_ROOT / "3_Output"
AUDIT_DIR = OUTPUT_DIR / "Audit"
FIGURES_DIR = OUTPUT_DIR / "Figures"
LOGS_DIR = OUTPUT_DIR / "Logs"
TABLES_DIR = OUTPUT_DIR / "Tables"

for path in [CLEANED_DIR, AUDIT_DIR, FIGURES_DIR, LOGS_DIR, TABLES_DIR]:
    path.mkdir(parents=True, exist_ok=True)
