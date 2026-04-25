"""
04_build_panel_dataset.py
==============================
Converts the base cross-sectional district data into a district-year panel (discrete-time hazard).
- Year of first-ever all-source adoption determines exit from the "at-risk" sample.
- Excludes own R1 winners.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANALYSIS_DATASET, ensure_dirs

def build_annual_panel(cross_section_df, start_year=2015, end_year=2024):
    """
    Constructs an annual panel from the district cross-section.
    Requires:
    - nces_id: District ID
    - year_first_awarded: First year the district won any ESB (WRI data)
    - is_r1_winner: Indicator for CSB Round 1 winner
    """
    print(f"Building annual discrete-time panel from {start_year} to {end_year}...")
    
    # Expand to district-year panel
    df_list = []
    for year in range(start_year, end_year + 1):
        temp = cross_section_df.copy()
        temp['year'] = year
        df_list.append(temp)
        
    panel = pd.concat(df_list, ignore_index=True)
    
    # 1. Outcome Variable (Y_it)
    # 1 if they are awarded their *first* ESB in this exact calendar year
    panel['Y_first_adopted'] = (panel['year_first_awarded'] == panel['year']).astype(int)
    
    # Also add outcomes for operating and delivered
    if 'year_first_operating' in panel.columns:
        panel['Y_first_operating'] = (panel['year_first_operating'] == panel['year']).astype(int)
    if 'year_first_delivered' in panel.columns:
        panel['Y_first_delivered'] = (panel['year_first_delivered'] == panel['year']).astype(int)
    
    # 2. At-risk constraint: flags for hazard model (drop years strictly after a district's first event year)
    # (Since it's a hazard model of first event, subsequent years are deterministic/not at risk)
    if 'year_first_awarded' in panel.columns:
        panel['hazard_keep_awarded'] = (panel['year'] <= panel['year_first_awarded'].fillna(9999)).astype(int)
        
    if 'year_first_operating' in panel.columns:
        panel['hazard_keep_operating'] = (panel['year'] <= panel['year_first_operating'].fillna(9999)).astype(int)
        
    if 'year_first_delivered' in panel.columns:
        panel['hazard_keep_delivered'] = (panel['year'] <= panel['year_first_delivered'].fillna(9999)).astype(int)
    
    # 3. Instrument constraint
    # Own R1 winners are dropped from the *estimating* sample later, so we flag them here.
    if 'is_r1_winner' in panel.columns:
        panel['exclude_own_r1_winner'] = panel['is_r1_winner'].fillna(0).astype(int)
    
    print(f"Panel size: {len(panel)} district-years")
    return panel

if __name__ == "__main__":
    # Load the base cross-section
    base_file = Path("1_Data/Cleaned/analysis_dataset.csv")
    if not base_file.exists():
        print(f"Warning: {base_file} not found. Please run 01_build_analysis_dataset.py first.")
        sys.exit(1)
        
    df_base = pd.read_csv(base_file, dtype={'nces_id': str})
    
    # Ensure expected columns are mapped if they have different names in the raw dataset
    if 'IV_Z_R1' in df_base.columns and 'is_r1_winner' not in df_base.columns:
        df_base['is_r1_winner'] = df_base['IV_Z_R1']
        
    # Build panel from 2018 to 2024
    if 'year_first_awarded' not in df_base.columns:
        print("Error: 'year_first_awarded' not found in base dataset.")
        print("Please ensure it is properly exported from 01_build_analysis_dataset.py.")
        sys.exit(1)
    
    panel = build_annual_panel(df_base, start_year=2018, end_year=2024)
    
    out_file = Path("1_Data/Cleaned/analysis_panel_dataset.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_file, index=False)
    print(f"Successfully saved panel dataset to {out_file}")
