import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def print_divider(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def compute_balance(df, group_col, test_vars):
    """Computes means and t-test p-values for two groups (col == 1 vs col == 0)"""
    g1 = df[df[group_col] == 1]
    g0 = df[df[group_col] == 0]
    
    results = []
    
    for var in test_vars:
        if var not in df.columns:
            continue
            
        m1 = g1[var].mean()
        m0 = g0[var].mean()
        
        # t-test handles NaNs by dropping them
        valid1 = g1[var].dropna()
        valid0 = g0[var].dropna()
        
        stat, pval = stats.ttest_ind(valid1, valid0, equal_var=False)
        
        results.append({
            'Variable': var,
            'Winner_Mean': m1,
            'Loser_Mean': m0,
            'Diff': m1 - m0,
            'T-stat': stat,
            'P-value': pval
        })
        
    return pd.DataFrame(results)

def main():
    data_path = Path("1_Data/Cleaned/analysis_dataset_spatial.csv")
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
        
    df = pd.read_csv(data_path, dtype={'nces_id': str})
    
    # Define applicants as either winners or losers
    df['is_r1_applicant'] = ((df['IV_Z_R1'] == 1) | (df['IS_R1_LOSER'] == 1)).astype(int)
    
    apps = df[df['is_r1_applicant'] == 1].copy()
    
    print_divider("Overall Applicant Balance")
    print(f"Total Applicants: {len(apps)}")
    print(f"Winners: {apps['IV_Z_R1'].sum()}")
    print(f"Losers: {len(apps) - apps['IV_Z_R1'].sum()}")
    
    test_vars = [
        'enrollment', 'median_income', 'poverty_rate', 'pct_white', 'pm25', 
        'priority_r1', 'is_pre_r1_adopter',
        'w6_P_t_minus_1', 'w6_priority_r1' # Can check spatial baseline characteristics
    ]
    
    # 1. Unconditional Balance among all applicants
    res_uncond = compute_balance(apps, 'IV_Z_R1', test_vars)
    print("\nUnconditional Balance (Winners vs Losers):")
    print(res_uncond.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # 2. Conditional Balance (among non-priority vs priority)
    print_divider("Conditional Balance (Non-Priority Only)")
    apps_non_priority = apps[apps['priority_r1'] == 0].copy()
    res_np = compute_balance(apps_non_priority, 'IV_Z_R1', test_vars)
    print(f"Sample Size: {len(apps_non_priority)}")
    print(res_np.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    print_divider("Conditional Balance (Priority Only)")
    apps_priority = apps[apps['priority_r1'] == 1].copy()
    res_p = compute_balance(apps_priority, 'IV_Z_R1', test_vars)
    print(f"Sample Size: {len(apps_priority)}")
    print(res_p.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Output to tables
    out_dir = Path("3_Output/Tables/Preliminaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    res_uncond.to_csv(out_dir / "r1_applicant_unconditional_balance.csv", index=False)
    res_np.to_csv(out_dir / "r1_applicant_nonpriority_balance.csv", index=False)
    res_p.to_csv(out_dir / "r1_applicant_priority_balance.csv", index=False)
    
    print(f"\nTables written to {out_dir}")

if __name__ == "__main__":
    main()
