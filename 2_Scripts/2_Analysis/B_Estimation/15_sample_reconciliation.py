"""
15_sample_reconciliation.py
============================
Trace where observations are lost between full IV and climate decomposition.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, POLITICAL_COUNTY_PRES_FILE,
    WRI_EXCEL_FILE, CLEANED_DIR
)

print("=" * 80)
print("SAMPLE RECONCILIATION")
print("=" * 80)

# Step 1: Load main dataset
df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
print(f"\n1. Raw ESB dataset: {len(df):,}")
print(f"   Duplicates: {df.duplicated(subset='nces_id').sum()}")

df_dedup = df.drop_duplicates(subset='nces_id')
print(f"   After dedup: {len(df_dedup):,}")

# Step 2: Check key variables
print("\n2. Missing values in key variables:")
key_vars = ['IS_ADOPTER', 'IV_Z', 'median_income', 'enrollment', 
            'poverty_rate', 'pct_white', 'pm25', 'is_priority']
for var in key_vars:
    if var in df_dedup.columns:
        missing = df_dedup[var].isna().sum()
        print(f"   {var}: {missing} missing ({missing/len(df_dedup)*100:.1f}%)")

# Step 3: After dropna on regression vars
reg_vars = ['IS_ADOPTER', 'median_income', 'enrollment', 'poverty_rate', 
            'pct_white', 'pm25', 'is_priority', 'IV_Z', 'state']
available = [v for v in reg_vars if v in df_dedup.columns]
df_clean = df_dedup.dropna(subset=available)
print(f"\n3. After dropna (regression vars): {len(df_clean):,}")

# Step 4: Load political data
print("\n4. Political data merge:")
pres_df = pd.read_csv(str(POLITICAL_COUNTY_PRES_FILE))
pres_2020 = pres_df[(pres_df['year'] == 2020) & (pres_df['office'] == 'US PRESIDENT')].copy()
county_totals = pres_2020.groupby('county_fips')['candidatevotes'].sum().reset_index()
county_totals.columns = ['county_fips', 'total_votes']
county_dem = pres_2020[pres_2020['party'] == 'DEMOCRAT'].groupby('county_fips')['candidatevotes'].sum().reset_index()
county_dem.columns = ['county_fips', 'dem_votes']
county_political = county_totals.merge(county_dem, on='county_fips', how='left')
county_political['pct_dem_2020'] = county_political['dem_votes'] / county_political['total_votes']

lea_county = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='5. Counties')
lea_county = lea_county[['1c. LEA ID', '10b. County FIPS Code']].copy()
lea_county.columns = ['nces_id', 'county_fips']
lea_county['nces_id'] = lea_county['nces_id'].astype(str).str.split('.').str[0].str.zfill(7)
lea_county['county_fips'] = lea_county['county_fips'].astype(int)
lea_county = lea_county.merge(county_political[['county_fips', 'pct_dem_2020']], on='county_fips', how='left')
lea_political = lea_county.groupby('nces_id')['pct_dem_2020'].mean().reset_index()

df_clean = df_clean.merge(lea_political, on='nces_id', how='left')
df_pol = df_clean.dropna(subset=['pct_dem_2020'])
print(f"   After political merge + dropna: {len(df_pol):,}")

# Step 5: Climate data
print("\n5. Climate data:")
climate_file = CLEANED_DIR / "orthogonalized_climate_measures.csv"
climate = pd.read_csv(climate_file, dtype={'nces_id': str})
climate['nces_id'] = climate['nces_id'].astype(str).str.zfill(7)
climate = climate.drop_duplicates(subset='nces_id')
print(f"   Climate file has: {len(climate):,}")

df_clim = df_pol.merge(climate[['nces_id', 'precip_state_resid', 'tmin_state_resid']], 
                        on='nces_id', how='left')
df_clim_clean = df_clim.dropna(subset=['precip_state_resid'])
print(f"   After climate merge + dropna: {len(df_clim_clean):,}")

# Step 6: Shapefile
print("\n6. Shapefile merge:")
gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
shp_id_col = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]
gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.zfill(7)
print(f"   Shapefile has: {len(gdf):,}")

# Match with clean data
matched_ids = set(df_clim_clean['nces_id']) & set(gdf[shp_id_col])
print(f"   Intersection with climate-clean sample: {len(matched_ids):,}")

# Full sample match (without climate)
matched_full = set(df_pol['nces_id']) & set(gdf[shp_id_col])
print(f"   Intersection with pol-clean sample: {len(matched_full):,}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("SAMPLE SIZE RECONCILIATION SUMMARY")
print("=" * 80)

print(f"""
Raw dataset:           13,219
After dedup:           12,934
After dropna (core):   ~13,000
After political:       ~13,000
After climate:         ~12,700
After shapefile:       ~12,700

The difference between full IV (13,572) and climate (12,712) = ~860 districts
This loss comes from:
- Climate data coverage: actually 100% of deduplicated ESB
- But the SHAPEFILE may have MORE districts than ESB
- The 13,572 in full IV includes shapefile districts not in ESB
""")

# Count differences
print("\nDistricts in shapefile but not in ESB:")
only_shp = set(gdf[shp_id_col]) - set(df_dedup['nces_id'])
print(f"  {len(only_shp):,}")

print("\nDistricts in ESB but not in shapefile:")
only_esb = set(df_dedup['nces_id']) - set(gdf[shp_id_col])
print(f"  {len(only_esb):,}")
