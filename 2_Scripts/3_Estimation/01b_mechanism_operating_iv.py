"""
01b_mechanism_operating_iv.py
=============================
Executes the first stage and 2SLS IV estimation on the discrete-time annual panel.

Panel covers 2018-2024.
DV: Y_first_operating
Treatment (endogenous): w6_P_operating_t_minus_1 (share of neighbors operating by end of t-1)
Instrument: w6_Z_t_minus_1 (share of neighbors winning R1 by end of t-1; 0 before 2023)
Baseline control: w6_P_operating_preR1 (w6_P_operating_t_minus_1 at start of 2022)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

warnings.filterwarnings("ignore")

def print_divider(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def main():
    panel_file = Path("1_Data/Cleaned/analysis_panel_dataset_spatial.csv")
    if not panel_file.exists():
        print(f"Missing {panel_file}")
        sys.exit(1)
        
    df = pd.read_csv(panel_file, dtype={'nces_id': str})
    
    # Extract baseline pre-R1 neighbor adoption (End of 2021 == P_t_minus_1 for year 2022)
    pre_r1_adopters = df[df['year'] == 2022][['nces_id', 'w6_P_operating_t_minus_1']].rename(
        columns={'w6_P_operating_t_minus_1': 'w6_P_operating_preR1'}
    )
    df = df.merge(pre_r1_adopters, on='nces_id', how='left')
    
    # 1. Sample Restrictions
    initial_n = len(df)
    
    # Filter for discrete-time hazard specific to Awarded
    df = df[df['hazard_keep_awarded'] == 1]
    
    # Drop own R1 winners from estimating sample
    df = df[df['exclude_own_r1_winner'] == 0]
    non_r1_n = len(df)
    
    print_divider(f"Sample Construction")
    print(f"Initial district-years: {initial_n:,}")
    print(f"After excluding own R1 winners: {non_r1_n:,} ({initial_n - non_r1_n:,} dropped)")

    # 2. Controls Prep
    df['log_enroll'] = np.log(df['enrollment'].clip(lower=1))
    df['log_income'] = np.log(df['median_income'].clip(lower=1))
    
    # Handle missing covariates by replacing with mean (standard for these analyses with FEs)
    covars = ['log_enroll', 'log_income', 'poverty_rate', 'pct_white', 'pm25', 'priority_r1']
    for c in covars:
        df[c] = df[c].fillna(df[c].mean())
        
    # Drop rows missing state or DV
    df = df.dropna(subset=['Y_first_operating', 'w6_P_operating_t_minus_1', 'w6_Z_t_minus_1', 'w6_P_operating_preR1', 'state'])
    print(f"Final Estimation Sample: {len(df):,} district-years\n")
    
    # 3. Additional Diagnostics requests
    print_divider("Sample Diagnostics")
    at_risk_by_year = df.groupby('year').size()
    print("At-Risk Districts by Year:")
    print(at_risk_by_year.to_string())
    
    inst_support = df.groupby('year')['w6_Z_t_minus_1'].describe()[['mean', 'std', 'min', 'max']]
    print("\nInstrument Support (w6_Z_t_minus_1) among At-Risk Districts by Year:")
    print(inst_support.to_string())
        
    # State and Year Fixed effects (State-by-year caused SVD convergence errors)
    state_dummies = pd.get_dummies(df['state'], drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(df['year'].astype(int), prefix='yr', drop_first=True, dtype=float)
    
    # Create Time-Varying EXCLUDED INSTRUMENTS (Converted to Raw Count instead of Share)
    # w6_* variables are shares from 6 neighbors, so * 6 gives count
    df['Z_x_2023'] = (df['w6_Z_t_minus_1'] * 6) * (df['year'] == 2023).astype(float)
    df['Z_x_2024'] = (df['w6_Z_t_minus_1'] * 6) * (df['year'] == 2024).astype(float)
    
    df['count_P_preR1'] = df['w6_P_operating_preR1'] * 6
    
    # Covariate Matrix
    X_cols = pd.concat([
        df[['log_enroll', 'log_income', 'poverty_rate', 'pct_white', 'pm25', 'priority_r1']], 
        df[['count_P_preR1']], 
        state_dummies,
        year_dummies
    ], axis=1)
    X_cols = sm.add_constant(X_cols, has_constant='add')
    
    Y = df['Y_first_operating'].astype(float)
    P_endog = (df['w6_P_operating_t_minus_1'] * 6).astype(float)
    P_endog.name = 'count_P_operating_t_minus_1'
    Z_instr = df[['Z_x_2023', 'Z_x_2024']].astype(float)
    
    # ----------------------------------------------------
    print_divider("1) NAIVE OLS REVEALING ENDOGENEITY")
    # Regress Y ~ P_endog + X
    exog_ols = pd.concat([P_endog, X_cols], axis=1)
    ols_res = sm.OLS(Y, exog_ols).fit(cov_type='cluster', cov_kwds={'groups': df['state']})
    print(f"Endogenous Peer Estimate (OLS):  {ols_res.params['count_P_operating_t_minus_1']:.4f} (SE: {ols_res.bse['count_P_operating_t_minus_1']:.4f}, p={ols_res.pvalues['count_P_operating_t_minus_1']:.4f})")
    
    # ----------------------------------------------------
    print_divider("2) FIRST STAGE: INSTRUMENT RELEVANCE")
    # Regress P_endog ~ Z_instr + X
    exog_first = pd.concat([Z_instr, X_cols], axis=1)
    fs_res = sm.OLS(P_endog, exog_first).fit(cov_type='cluster', cov_kwds={'groups': df['state']})
    print(f"w6_Z_t_minus_1 x (Year == 2023):  {fs_res.params['Z_x_2023']:.4f} (SE: {fs_res.bse['Z_x_2023']:.4f}, p={fs_res.pvalues['Z_x_2023']:.4f})")
    print(f"w6_Z_t_minus_1 x (Year == 2024):  {fs_res.params['Z_x_2024']:.4f} (SE: {fs_res.bse['Z_x_2024']:.4f}, p={fs_res.pvalues['Z_x_2024']:.4f})")
    
    # F-stat
    hyp_test = fs_res.f_test(['Z_x_2023 = 0', 'Z_x_2024 = 0'])
    fval = float(hyp_test.fvalue)
    print(f"First Stage F-stat on Zs:     {fval:.2f}")
    if fval < 10:
        print(" -> Warning: Weak instrument! (F < 10)")
    else:
        print(" -> Strong instrument! (F >= 10)")
        
    # ----------------------------------------------------
    print_divider("3) REDUCED FORM: ITT")
    # Regress Y ~ Z_instr + X
    exog_rf = pd.concat([Z_instr, X_cols], axis=1)
    rf_res = sm.OLS(Y, exog_rf).fit(cov_type='cluster', cov_kwds={'groups': df['state']})
    print(f"Reduced Form ITT x 2023:      {rf_res.params['Z_x_2023']:.4f} (SE: {rf_res.bse['Z_x_2023']:.4f}, p={rf_res.pvalues['Z_x_2023']:.4f})")
    print(f"Reduced Form ITT x 2024:      {rf_res.params['Z_x_2024']:.4f} (SE: {rf_res.bse['Z_x_2024']:.4f}, p={rf_res.pvalues['Z_x_2024']:.4f})")

    # ----------------------------------------------------
    print_divider("4) SECOND STAGE: 2SLS IV ESTIMATE")
    try:
        iv_res = IV2SLS(
            dependent=Y,
            exog=X_cols,
            endog=P_endog,
            instruments=Z_instr
        ).fit(cov_type='clustered', clusters=df['state'])
        
        print(iv_res.summary)
        
    except Exception as e:
        print(f"IV2SLS Failed: {e}")
        
if __name__ == '__main__':
    main()
