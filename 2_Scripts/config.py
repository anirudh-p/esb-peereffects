"""
config.py  —  Post-Temporal branch
All raw-data paths and output directories referenced by analysis scripts.
"""

from pathlib import Path

# ── Repo root ──────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent           # c:\BC PhD\Research\Peer-Effects and Adoption
DATA   = ROOT / "1_Data"
RAW    = DATA / "Raw"
CLEAN  = DATA / "Cleaned"

# ── Raw data ───────────────────────────────────────────────────────────────────
WRI_DIR         = RAW / "WRI"
WRI_DISTRICT    = WRI_DIR / "wri_data.csv"           # sheet "1. District-level data"  (19,517 rows)
WRI_BUS_EXCEL   = WRI_DIR / "ESB_adoption_dataset_v9_update_june_2025.xlsx"
CSB_REBATES     = WRI_DIR / "CSB_Rebates.xlsx"       # R1 (2022) + R3 (2023) lottery winners
CSB_GRANTS      = WRI_DIR / "CSB_Grants.xlsx"        # R2 (2023) competitive grants — NOT lottery
CSB_APPLICANTS  = WRI_DIR / "CSBP Applicants waitlisted and rejected_11.18.25.xlsx"

SHP_FILE = (RAW / "Spatial" / "EDGE_SCHOOLDISTRICT_TL21_SY2021"
               / "schooldistrict_sy2021_tl21.shp")

POL_FILE = RAW / "Political" / "dataverse_files" / "countypres_2000-2024.csv"

# ── Outputs ────────────────────────────────────────────────────────────────────
LOGS_DIR    = ROOT / "3_Output" / "Logs"
TABLES_DIR  = ROOT / "3_Output" / "Tables"
FIGURES_DIR = ROOT / "3_Output" / "Figures"

# Analysis-ready dataset (written by 01_build_analysis_dataset.py)
ANALYSIS_DATASET = CLEAN / "analysis_dataset.csv"

# Spatial dataset with KNN lags (written by 02_build_spatial_weights.py)
SPATIAL_DATASET  = CLEAN / "analysis_dataset_spatial.csv"


def ensure_dirs():
    for d in [CLEAN, LOGS_DIR, TABLES_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
