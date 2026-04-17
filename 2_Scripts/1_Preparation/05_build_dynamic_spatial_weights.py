"""
05_build_dynamic_spatial_weights.py
====================================
Constructs dynamic spatial components ($P_{i,t-1}$ and $Z_{i,t-1}$) over the district-year panel.
It preserves the static geographic neighbors but applies them to the time-varying adoption and lottery statuses.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ensure_dirs

def build_dynamic_spatial_lags(df_panel, df_spatial, W_matrix, prefix="w6"):
    """
    Given a district-year panel and a static W matrix (row-standardized or binary),
    computes the spatial lags for adoption ($P_{i,t-1}$) and instrument ($Z_{i,t-1}$).
    """
    print(f"Calculating dynamic spatial lags ({prefix})...")
    
    # Read the exact NCES ID order used to construct W
    order_file = Path("1_Data/Cleaned/spatial_nces_order.csv")
    if not order_file.exists():
        print("Missing spatial_nces_order.csv. Run 02_build_spatial_weights.py first.")
        sys.exit(1)
        
    df_order = pd.read_csv(order_file, dtype={'nces_id': str})
    N = len(df_order)
    nces_order = df_order['nces_id'].values
    
    # Merge the required features onto the strictly ordered dataframe
    df_spatial = df_order.merge(df_spatial, on='nces_id', how='left')
    
    # Extract static attributes needed for vectors in EXACT order of W matrix
    # Merge year_first_awarded and year_first_operating from the base analysis dataset
    base_df = pd.read_csv("1_Data/Cleaned/analysis_dataset.csv", dtype={'nces_id': str})
    
    # Drop existing outcome variables if they exist to avoid duplication
    cols_to_drop = ['year_first_awarded', 'year_first_operating', 'IV_Z_R1']
    df_spatial = df_spatial.drop(columns=[col for col in cols_to_drop if col in df_spatial.columns])
    
    df_spatial = df_spatial.merge(base_df[['nces_id', 'year_first_awarded', 'year_first_operating', 'IV_Z_R1']], on='nces_id', how='left')
        
    year_first_awarded = df_spatial['year_first_awarded'].fillna(9999).values
    year_first_operating = df_spatial['year_first_operating'].fillna(9999).values
    is_r1_winner = df_spatial['IV_Z_R1'].fillna(0).values
    
    lags_list = []
    years = sorted(df_panel['year'].unique())
    
    for t in years:
        # P_{t-1}: Adopted by end of t-1
        P_t_minus_1 = (year_first_awarded <= (t - 1)).astype(float)
        # P_operating_{t-1}: Bus delivered and operating by end of t-1
        P_operating_t_minus_1 = (year_first_operating <= (t - 1)).astype(float)
        
        # Z_{t-1}: Won R1 by end of t-1 (R1 occurred in 2022)
        if t - 1 >= 2022:
            Z_t_minus_1 = is_r1_winner.astype(float)
        else:
            Z_t_minus_1 = np.zeros(N)
            
        # Compute spatial lags using sparse dot product
        w_P = W_matrix.dot(P_t_minus_1)
        w_P_operating = W_matrix.dot(P_operating_t_minus_1)
        w_Z = W_matrix.dot(Z_t_minus_1)
        
        temp = pd.DataFrame({
            'nces_id': nces_order,
            'year': t,
            f'{prefix}_P_t_minus_1': w_P,
            f'{prefix}_P_operating_t_minus_1': w_P_operating,
            f'{prefix}_Z_t_minus_1': w_Z
        })
        lags_list.append(temp)
        
    df_lags = pd.concat(lags_list, ignore_index=True)
    
    # Merge back into panel
    df_panel = df_panel.merge(df_lags, on=['nces_id', 'year'], how='left')
    print("Phase 2 dynamic spatial lags complete.")
    return df_panel

if __name__ == "__main__":
    from scipy.sparse import load_npz
    
    panel_file = Path("1_Data/Cleaned/analysis_panel_dataset.csv")
    if not panel_file.exists():
        print(f"Warning: {panel_file} not found. Run 04_build_panel_dataset.py first.")
        sys.exit(1)
        
    df_panel = pd.read_csv(panel_file, dtype={'nces_id': str})
    
    base_m_file = Path("1_Data/Cleaned/analysis_dataset_spatial.csv")
    if not base_m_file.exists():
        print("Warning: Base spatial dataset not found.")
        sys.exit(1)
    
    df_spatial = pd.read_csv(base_m_file, dtype={'nces_id': str})
    
    # We must restrict to the exact 13,075 districts that actually entered the W matrix
    # The spatial script generates w{K}_is_r1_winner only for non-null shapefile matches.
    # We can filter to districts where that lag is not NaN.
    df_spatial_matched = df_spatial[df_spatial['w6_IV_Z_R1'].notna()].copy()
    
    K_VALUES = [4, 6, 8, 10, 15]
    for K in K_VALUES:
        weights_file = Path(f"1_Data/Cleaned/knn_weights_k{K}.npz")
        try:
            W = load_npz(weights_file)
            print(f"Loaded W matrix for K={K}")
            df_panel = build_dynamic_spatial_lags(df_panel, df_spatial_matched, W, prefix=f"w{K}")
        except OSError:
            print(f"Weights file {weights_file} not found. Skipping K={K}.")
            
    R_VALUES = [15, 30, 60]
    for R in R_VALUES:
        weights_file = Path(f"1_Data/Cleaned/radius_weights_r{R}.npz")
        try:
            W = load_npz(weights_file)
            print(f"Loaded W matrix for R={R}")
            df_panel = build_dynamic_spatial_lags(df_panel, df_spatial_matched, W, prefix=f"r{R}")
        except OSError:
            print(f"Weights file {weights_file} not found. Skipping R={R}.")
            
    out_file = Path("1_Data/Cleaned/analysis_panel_dataset_spatial.csv")
    df_panel.to_csv(out_file, index=False)
    print(f"Saved dynamic spatial panel to {out_file}")
    
    # Export to .dta for Stata estimation
    dta_out_file = Path("1_Data/Cleaned/analysis_panel_dataset_spatial.dta")
    df_panel.to_stata(dta_out_file, write_index=False, version=118)
    print(f"Saved dynamic spatial panel to STATA .dta at {dta_out_file}")

