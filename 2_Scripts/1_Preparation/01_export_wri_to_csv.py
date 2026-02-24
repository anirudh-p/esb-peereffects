"""
01_export_wri_to_csv.py
======================
Export WRI Excel sheet to CSV to avoid forrtl crashes.
This is a prerequisite step that creates a CSV backup of the WRI data.

Inputs:
    - Raw/WRI/ESB_adoption_dataset_v9_update_june_2025.xlsx

Outputs:
    - Raw/WRI/wri_data.csv

Dependencies: None (first script in pipeline)

Author: [Your Name]
Last Updated: November 30, 2025
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_WRI_DIR, WRI_CSV_FILE

EXCEL_FILE = RAW_WRI_DIR / 'ESB_adoption_dataset_v9_update_june_2025.xlsx'
OUTPUT_CSV = WRI_CSV_FILE

print(f"Attempting to export {EXCEL_FILE} to CSV...")
print("(This may crash with forrtl error - if so, manually export in Excel)")

try:
    # Try different engines
    for engine in [None, 'openpyxl']:
        try:
            print(f"\nTrying engine={engine}...")
            df = pd.read_excel(EXCEL_FILE, sheet_name='1. District-level data', engine=engine)
            print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Save to CSV
            df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
            print(f"\n✓ SUCCESS: Saved to {OUTPUT_CSV}")
            print(f"  Size: {len(df)} districts")
            print(f"  Columns: {df.columns.tolist()[:5]}...")
            sys.exit(0)
            
        except Exception as e:
            print(f"  Failed with {engine}: {str(e)[:100]}")
            continue
    
    print("\n✗ All engines failed")
    print("\nMANUAL EXPORT REQUIRED:")
    print("1. Open ESB_adoption_dataset_v9_update_june_2025.xlsx in Excel")
    print("2. Go to sheet '1. District-level data'")
    print(f"3. Save As → CSV → {OUTPUT_CSV}")
    sys.exit(1)
    
except KeyboardInterrupt:
    print("\n\nInterrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ FATAL ERROR: {e}")
    sys.exit(1)
