"""
14_dropped_districts_analysis.py
================================
Analyze which districts are dropped when merging orthogonalized climate data.
Identify systematic patterns in data loss.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import ESB_FULL_ANALYSIS, CLEANED_DIR

print("=" * 80)
print("DROPPED DISTRICTS ANALYSIS")
print("=" * 80)

# Load main dataset
df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
df = df.drop_duplicates(subset='nces_id')
print(f"\nFull dataset: {len(df):,} districts")

# Load orthogonalized climate
climate_file = CLEANED_DIR / "orthogonalized_climate_measures.csv"
if climate_file.exists():
    climate = pd.read_csv(climate_file, dtype={'nces_id': str})
    climate['nces_id'] = climate['nces_id'].astype(str).str.zfill(7)
    climate = climate.drop_duplicates(subset='nces_id')
    print(f"Climate data: {len(climate):,} districts")
else:
    print("Climate file not found!")
    sys.exit(1)

# Merge and identify dropped
df['has_climate'] = df['nces_id'].isin(climate['nces_id'])
dropped = df[~df['has_climate']].copy()
kept = df[df['has_climate']].copy()

print(f"\nDistricts WITH climate data: {len(kept):,}")
print(f"Districts DROPPED (no climate): {len(dropped):,}")

# ==============================================================================
# COMPARE DROPPED vs KEPT
# ==============================================================================

print("\n" + "=" * 80)
print("COMPARISON: DROPPED vs KEPT DISTRICTS")
print("=" * 80)

comparison_vars = ['IS_ADOPTER', 'IV_Z', 'IS_APPLICANT', 'median_income', 
                   'enrollment', 'poverty_rate', 'pct_white', 'pm25']

print(f"\n{'Variable':<20} {'Dropped Mean':>15} {'Kept Mean':>15} {'Diff':>10}")
print("-" * 60)

for var in comparison_vars:
    if var not in df.columns:
        continue
    dropped_mean = dropped[var].mean()
    kept_mean = kept[var].mean()
    diff = dropped_mean - kept_mean
    print(f"{var:<20} {dropped_mean:>15.4f} {kept_mean:>15.4f} {diff:>10.4f}")

# ==============================================================================
# STATE-LEVEL ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("STATE-LEVEL DROP RATES")
print("=" * 80)

state_stats = df.groupby('state').agg({
    'has_climate': ['sum', 'count', 'mean']
}).round(3)
state_stats.columns = ['kept', 'total', 'kept_pct']
state_stats['dropped'] = state_stats['total'] - state_stats['kept']
state_stats['dropped_pct'] = 1 - state_stats['kept_pct']
state_stats = state_stats.sort_values('dropped_pct', ascending=False)

print(f"\nStates with highest drop rates:")
print(state_stats[state_stats['dropped_pct'] > 0.05].head(20))

# Count districts by applicant status
print("\n" + "=" * 80)
print("APPLICANT STATUS IN DROPPED DISTRICTS")
print("=" * 80)

if 'IS_APPLICANT' in df.columns:
    dropped_applicants = dropped['IS_APPLICANT'].sum()
    kept_applicants = kept['IS_APPLICANT'].sum()
    total_applicants = df['IS_APPLICANT'].sum()
    
    print(f"\nTotal applicants: {total_applicants:.0f}")
    print(f"Applicants in dropped: {dropped_applicants:.0f} ({dropped_applicants/total_applicants*100:.1f}%)")
    print(f"Applicants in kept: {kept_applicants:.0f} ({kept_applicants/total_applicants*100:.1f}%)")
    
    # Winners distribution
    dropped_winners = dropped['IV_Z'].sum()
    kept_winners = kept['IV_Z'].sum()
    total_winners = df['IV_Z'].sum()
    
    print(f"\nTotal lottery winners: {total_winners:.0f}")
    print(f"Winners in dropped: {dropped_winners:.0f} ({dropped_winners/total_winners*100:.1f}%)")
    print(f"Winners in kept: {kept_winners:.0f} ({kept_winners/total_winners*100:.1f}%)")

# ==============================================================================
# ADOPTER STATUS IN DROPPED
# ==============================================================================

print("\n" + "=" * 80)
print("ADOPTER STATUS")
print("=" * 80)

dropped_adopters = dropped['IS_ADOPTER'].sum()
kept_adopters = kept['IS_ADOPTER'].sum()
total_adopters = df['IS_ADOPTER'].sum()

print(f"\nTotal adopters: {total_adopters:.0f}")
print(f"Adopters in dropped: {dropped_adopters:.0f} ({dropped_adopters/total_adopters*100:.1f}%)")
print(f"Adopters in kept: {kept_adopters:.0f} ({kept_adopters/total_adopters*100:.1f}%)")

# Adoption rate comparison
print(f"\nAdoption rate in dropped: {dropped['IS_ADOPTER'].mean()*100:.2f}%")
print(f"Adoption rate in kept: {kept['IS_ADOPTER'].mean()*100:.2f}%")

# ==============================================================================
# WHY NO CLIMATE DATA?
# ==============================================================================

print("\n" + "=" * 80)
print("DIAGNOSING MISSING CLIMATE DATA")
print("=" * 80)

# Check if climate vars exist in main df
climate_vars = ['prcp_avg', 'tmin_avg', 'annual_precip', 'min_temp']
for var in climate_vars:
    if var in df.columns:
        missing = df[var].isna().sum()
        print(f"{var}: {missing} missing ({missing/len(df)*100:.1f}%)")

# Sample some dropped districts
print("\n\nSample dropped districts:")
sample = dropped[['nces_id', 'state', 'IS_ADOPTER', 'IS_APPLICANT', 'enrollment']].head(10)
print(sample.to_string())

# Check state distribution
print("\n\nDropped districts by state (top 10):")
print(dropped['state'].value_counts().head(10))

print("\n" + "=" * 80)
print("DATA QUALITY RECOMMENDATIONS")
print("=" * 80)

print("""
Based on this analysis:
1. If drops are RANDOM: Climate restriction is fine, just reduces power
2. If drops are SYSTEMATIC (certain states/sizes): May introduce selection bias
3. Alternative: Use different climate source that has better coverage
""")
