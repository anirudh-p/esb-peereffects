"""
04_esb_analysis_prep.py
========================
Merge master regression data with WRI agentic variables and priority status.
Creates final analysis dataset with covariates and runs balance checks.

Inputs:
    - Cleaned/esb_master_data_for_regression.csv (from script 03)
    - Raw/WRI/wri_data.csv (from script 01)
    - Raw/Other/CSBP Applicants waitlisted and rejected_11.18.25.xlsx

Outputs:
    - Cleaned/esb_full_analysis_dataset.csv
    - Tables/balance_table_stratified_v2.txt

Dependencies: 01_export_wri_to_csv.py, 03_data_prep_and_map.py

Author: [Your Name]
Last Updated: November 30, 2025
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ESB_MASTER_DATA, WRI_CSV_FILE, APPLICANT_FILE,
    ESB_FULL_ANALYSIS, TABLES_DIR, ensure_dirs_exist
)

# Ensure output directories exist
ensure_dirs_exist()

# Output paths
BALANCE_TABLE_FILE = TABLES_DIR / 'balance_table_stratified_v2.txt'

# --- 1. DATA LOADING ---

print("Loading Event Data (R1/R2/R3 Status)...")
df_events = pd.read_csv(str(ESB_MASTER_DATA), dtype={'NCES District ID': str})

print("Loading Applicant Priority Data...")
# We need the specific 'School District Prioritized' status from the application
df_app_raw = pd.read_excel(str(APPLICANT_FILE))
# Standardize ID for merge
df_app_raw['NCES District ID'] = df_app_raw['NCES District ID'].astype(str).str.split('.').str[0].str.zfill(7)

# Extract Priority Status
# Note: A district might appear multiple times if they applied to multiple rounds.
# We take the "maximum" priority status (if they were ever prioritized, count them as priority).
# Sorting ensures 'Yes' comes before 'No' or NaN if duplicates exist.
df_app_raw.sort_values('School District Prioritized', ascending=False, inplace=True)
df_priority = df_app_raw[['NCES District ID', 'School District Prioritized']].drop_duplicates(subset='NCES District ID')

print("Loading WRI Agentic Data (v9)...")
# Use CSV backup from script 01
print(f"  Using CSV: {WRI_CSV_FILE}")
df_wri = pd.read_csv(str(WRI_CSV_FILE))

# --- NEW FILTERING STEP WITH WATERFALL TRACKING ---
n_raw = len(df_wri)
print(f"\n--- DATA CLEANING STEP 1: FILTERING UNIVERSE ---")
print(f"Raw Universe Size: {n_raw}")

# Filter for Public School Districts: Traditional (Types 1, 2) AND Charters (Type 7)
# We exclude Admin Agencies (3,4,5,6,8,9) but INCLUDE charters
df_wri_public = df_wri[df_wri['1k. LEA type (number)'].isin([1, 2, 7])]
n_public = len(df_wri_public)
n_dropped_type = n_raw - n_public

print(f"Dropped Non-Public Entities (Admin Agencies, etc.): {n_dropped_type}")
print(f"Remaining Public Districts (including charters): {n_public}")

# Count charters for tracking
n_charters = len(df_wri_public[df_wri_public['1k. LEA type (number)'] == 7])
print(f"  Traditional districts: {n_public - n_charters}")
print(f"  Charter schools: {n_charters}")

# Use the filtered dataset moving forward
df_wri = df_wri_public.copy()


# --- 2. CLEANING & RENAMING WRI VARIABLES ---

column_map = {
    '1c. LEA ID': 'nces_id',
    '1b. Local Education Agency (LEA) or entity name': 'district_name',
    '1g. State': 'state',
    '1k. LEA type (number)': 'lea_type',  # ADD LEA TYPE
    '1p. Locale broad type (name)': 'urbanicity',
    '2a. Total number of buses': 'fleet_size',
    '3a. Number of ESBs committed': 'total_esbs_committed',
    '4b. Number of students in district': 'enrollment',
    '4f. Median household income': 'median_income',
    '4g. Percent of population below the poverty level': 'poverty_rate',
    '4u. Percent Hispanic or Latino (of any race) ': 'pct_hispanic',
    '4i. Percent race alone or multiracial: White': 'pct_white',
    '4k. Percent race alone or multiracial: Black or African American': 'pct_black',
    '5f. PM2.5 concentration': 'pm25',
}

available_cols = [c for c in column_map.keys() if c in df_wri.columns]
df_wri_clean = df_wri[available_cols].rename(columns=column_map)
df_wri_clean['nces_id'] = df_wri_clean['nces_id'].astype(str).str.split('.').str[0].str.zfill(7)

# Create charter flag
df_wri_clean['is_charter'] = (df_wri_clean['lea_type'] == 7).astype(int)
print(f"Created charter flag: {df_wri_clean['is_charter'].sum()} charters marked")

# --- 3. MERGING ---

print("Merging Datasets...")
# 1. Merge Universe with Event Data
df_analysis = pd.merge(df_wri_clean, df_events, left_on='nces_id', right_on='NCES District ID', how='left')

# 2. Merge with Priority Data
df_analysis = pd.merge(df_analysis, df_priority, left_on='nces_id', right_on='NCES District ID', how='left')

# Fill NaNs
fill_zeros = ['IS_ADOPTER', 'IS_APPLICANT', 'NUM_ADOPTED', 'IV_Z']
for col in fill_zeros:
    df_analysis[col] = df_analysis[col].fillna(0).astype(int)

# --- LOGIC FIX: BACKFILL APPLICANT STATUS ---
# The applicant file only contained losers. Winners are also applicants.
# If you adopted (IS_ADOPTER=1), you must have applied (IS_APPLICANT=1).
df_analysis.loc[df_analysis['IS_ADOPTER'] == 1, 'IS_APPLICANT'] = 1
# ---------------------------------------------

df_analysis['Funding Mechanism'] = df_analysis['Funding Mechanism'].fillna('None')

# Create Binary Priority Variable
# Normalize 'Yes'/'No'/'TRUE'/'FALSE'
def normalize_priority(x):
    if pd.isna(x): return 0
    s = str(x).upper()
    return 1 if s in ['YES', 'TRUE', '1'] else 0

df_analysis['is_priority'] = df_analysis['School District Prioritized'].apply(normalize_priority)

# Define "Lottery Loser" (Applicant who didn't win Lottery)
# Note: We restrict 'Winners' to Lottery Winners (IV_Z=1) to avoid conflating R2 winners
df_analysis['IS_LOTTERY_WINNER'] = df_analysis['IV_Z']
df_analysis['IS_LOTTERY_LOSER'] = ((df_analysis['IS_APPLICANT'] == 1) & (df_analysis['IV_Z'] == 0)).astype(int)


# --- FILTERING STEP 2: MISSING COVARIATES ---
# We check for missing data in key regression variables
key_vars = ['median_income', 'poverty_rate', 'enrollment', 'pm25', 'pct_white']
n_before_missing = len(df_analysis)
df_analysis_clean = df_analysis.dropna(subset=key_vars)
n_after_missing = len(df_analysis_clean)
n_dropped_missing = n_before_missing - n_after_missing

print(f"\n--- DATA CLEANING STEP 2: MISSING DATA ---")
print(f"Districts with Missing Covariates (Income, Race, etc.): {n_dropped_missing}")
print(f"Final Analytical Sample Size: {n_after_missing}")

# Use the fully clean dataset for analysis
df_analysis = df_analysis_clean.copy()


# --- 4. ANALYSIS & VALIDATION ---

covariates = ['median_income', 'poverty_rate', 'enrollment', 'pm25', 'pct_white', 'pct_black', 'pct_hispanic', 'fleet_size']

def run_balance_test(df, group_col, group1_label, group0_label):
    results = []
    for cov in covariates:
        # Drop missing values for this specific test
        subset = df.dropna(subset=[cov, group_col])
        if len(subset) == 0: continue
        
        g1 = subset[subset[group_col] == 1][cov]
        g0 = subset[subset[group_col] == 0][cov]
        
        if len(g1) < 2 or len(g0) < 2: continue 

        t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False)
        
        stars = ''
        if p_val < 0.01: stars = '***'
        elif p_val < 0.05: stars = '**'
        elif p_val < 0.1: stars = '*'
        
        results.append({
            'Variable': cov,
            f'Mean ({group1_label})': g1.mean(),
            f'Mean ({group0_label})': g0.mean(),
            'Diff': g1.mean() - g0.mean(),
            'P-Val': f"{p_val:.3f}{stars}"
        })
    return pd.DataFrame(results).round(2)

# --- 4A. THE "NAIVE" TEST (All Applicants) ---
print("\n--- NAIVE BALANCE TEST (All Applicants) ---")
# Sample: All Applicants
applicant_pool = df_analysis[df_analysis['IS_APPLICANT'] == 1].copy()
# Treatment: Winning the Lottery (IV_Z=1) vs Losing (IV_Z=0)
df_naive = run_balance_test(applicant_pool, 'IV_Z', 'Winners', 'Losers')
print(df_naive)


# --- 4B. THE "STRATIFIED" TEST (By Priority Status) ---
print("\n--- STRATIFIED BALANCE TEST (By Priority Status) ---")

with open(BALANCE_TABLE_FILE, 'w') as f:
    f.write("TABLE 2: RANDOMIZATION BALANCE CHECK (STRATIFIED)\n")
    f.write("=================================================\n")
    f.write(f"Sample Note: Public School Districts (Traditional + Charters, Types 1, 2, 7)\n")
    f.write(f"Universe N={n_public}. Applicant N={len(applicant_pool)}.\n\n")
    
    f.write("PANEL A: NAIVE (All Applicants)\n")
    f.write("-" * 60 + "\n")
    f.write(df_naive.to_string(index=False))
    f.write("\n\n")

    for status, label in [(1, "PRIORITY"), (0, "NON-PRIORITY")]:
        print(f"\nRunning Balance for {label} Applicants...")
        
        # Filter Sample to Just this Stratum
        sub_pool = applicant_pool[applicant_pool['is_priority'] == status]
        
        if len(sub_pool) > 10:
            df_strata = run_balance_test(sub_pool, 'IV_Z', 'Winners', 'Losers')
            print(df_strata)
            
            f.write(f"PANEL B.{status}: {label} APPLICANTS ONLY (N={len(sub_pool)})\n")
            f.write("-" * 60 + "\n")
            f.write(df_strata.to_string(index=False))
            f.write("\n\n")
        else:
            print(f"Skipping {label} - Sample too small (N={len(sub_pool)})")

print(f"\nBalance tables saved to {BALANCE_TABLE_FILE}")


# --- 5. REGRESSION ANALYSIS (OLS) ---

print("\nRunning OLS: Determinants of Adoption...")
df_reg = df_analysis.copy()
# Log transform skewed variables
for col in ['median_income', 'enrollment']:
    df_reg[f'log_{col}'] = np.log(df_reg[col] + 1)

# Create dummy variables for urbanicity (rural/urban status)
# Categories: City, Suburb, Town, Rural
print(f"\nUrbanicity distribution:")
if 'urbanicity' in df_reg.columns:
    print(df_reg['urbanicity'].value_counts())
    # Create dummy variables (drop first to avoid multicollinearity)
    urbanicity_dummies = pd.get_dummies(df_reg['urbanicity'], prefix='locale', drop_first=True)
    df_reg = pd.concat([df_reg, urbanicity_dummies], axis=1)
    # Get list of dummy column names
    locale_vars = [col for col in df_reg.columns if col.startswith('locale_')]
    print(f"Created dummy variables: {locale_vars}")
else:
    print("Warning: urbanicity variable not found in dataset")
    locale_vars = []

# FIX 1: Include all racial variables in Naive OLS
print("\nModel 1: Naive OLS (Full Demographics)")
formula1 = "IS_ADOPTER ~ log_median_income + poverty_rate + log_enrollment + pm25 + pct_white + pct_black + pct_hispanic"
model1 = smf.ols(formula1, data=df_reg).fit()
print(model1.summary().tables[1])

# Model 2: Adding Priority Status and Rural/Urban Controls
print("\nModel 2: OLS Controlling for Priority Status and Urbanicity")
locale_formula = ' + ' + ' + '.join(locale_vars) if locale_vars else ''
formula2 = f"IS_ADOPTER ~ log_median_income + poverty_rate + log_enrollment + pm25 + pct_white + pct_black + pct_hispanic + is_priority{locale_formula}"
model2 = smf.ols(formula2, data=df_reg).fit()
print(model2.summary().tables[1])

# FIX 2: Model 3 with Interaction Terms (Robustness Check)
print("\nModel 3: OLS with Priority × Poverty Interaction + Urbanicity (Robustness)")
formula3 = f"IS_ADOPTER ~ log_median_income + poverty_rate + log_enrollment + pm25 + pct_white + pct_black + pct_hispanic + is_priority + is_priority:poverty_rate{locale_formula}"
model3 = smf.ols(formula3, data=df_reg).fit()
print(model3.summary().tables[1])

# --- 6. EXPORT ---
df_analysis.to_csv(str(ESB_FULL_ANALYSIS), index=False)
print(f"\nFull Analysis Dataset Exported: {ESB_FULL_ANALYSIS}")
print(f"Final sample size: {len(df_analysis)} districts")
print(f"Charters: {df_analysis['is_charter'].sum()}")
print(f"Traditional: {(df_analysis['is_charter']==0).sum()}")