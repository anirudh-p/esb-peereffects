"""
02_extract_charter_flags.py
============================
Extract charter school flags from WRI dataset.
Creates a mapping file used in subsequent data preparation steps.

Inputs:
    - Raw/WRI/wri_data.csv (from script 01)

Outputs:
    - Cleaned/charter_flags.csv

Dependencies: 01_export_wri_to_csv.py

Author: [Your Name]
Last Updated: November 30, 2025
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WRI_CSV_FILE, CLEANED_DIR, ensure_dirs_exist

# Ensure output directory exists
ensure_dirs_exist()

print('Loading WRI data to extract charter flags...')
df = pd.read_csv(WRI_CSV_FILE)

print(f'Loaded {len(df)} rows')

# Extract charter flags
df['lea_id_str'] = df['1c. LEA ID'].astype(str).str.replace('.0', '', regex=False)
df_clean = df[df['lea_id_str'].str.isdigit()].copy()
df_clean['nces_id'] = df_clean['lea_id_str'].astype(int)

df_clean['is_charter'] = (
    (df_clean['1k. LEA type (number)'] == 7) | 
    (df_clean['1l. LEA type (name)'].str.contains('Charter', case=False, na=False))
).astype(int)

charter_flags = df_clean[['nces_id', 'is_charter']].drop_duplicates()

# Save to Cleaned directory
output_file = CLEANED_DIR / 'charter_flags.csv'
charter_flags.to_csv(output_file, index=False)

print(f'Saved {len(charter_flags)} charter flags to {output_file}')
print(f'Charters: {charter_flags["is_charter"].sum()}')
print('Done!')
