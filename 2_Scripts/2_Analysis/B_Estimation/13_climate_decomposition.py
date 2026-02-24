"""
13_climate_decomposition.py
===========================
Test if climate similarity matters BEYOND geographic proximity.

Uses orthogonalized (state-residualized) climate measures to build
climate peer networks, then runs horse-race with geographic peers.

Key Question: After controlling for geographic proximity, do districts
with similar within-state climate deviate have correlated adoption?
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import sys

# Force stdout to be unbuffered on Windows
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, POLITICAL_COUNTY_PRES_FILE,
    WRI_EXCEL_FILE, CLEANED_DIR, TABLES_DIR, LOGS_DIR, ensure_dirs_exist
)

from libpysal.weights import KNN
from libpysal.weights import lag_spatial
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors

ensure_dirs_exist()

print("=" * 80)
print("CLIMATE DECOMPOSITION: Does Climate Matter Beyond Geography?")
print("=" * 80)

# ==============================================================================
# LOAD DATA
# ==============================================================================

# Main dataset
df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
df = df.drop_duplicates(subset='nces_id')  # Ensure no duplicates
print(f"\nLoaded {len(df):,} districts")

# Orthogonalized climate measures - drop duplicates before merge
climate_ortho = pd.read_csv(CLEANED_DIR / "orthogonalized_climate_measures.csv",
                            dtype={'nces_id': str})
climate_ortho['nces_id'] = climate_ortho['nces_id'].astype(str).str.zfill(7)
climate_ortho = climate_ortho.drop_duplicates(subset='nces_id')  # Critical!
df = df.merge(climate_ortho[['nces_id', 'precip_state_resid', 'tmin_state_resid']], 
              on='nces_id', how='left')
print(f"With orthogonalized climate: {df['precip_state_resid'].notna().sum()}")

# Engineered features
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# Political data
print("\nLoading political data...")
pres_df = pd.read_csv(str(POLITICAL_COUNTY_PRES_FILE))
pres_2020 = pres_df[(pres_df['year'] == 2020) & (pres_df['office'] == 'US PRESIDENT')].copy()
county_totals = pres_2020.groupby('county_fips')['candidatevotes'].sum().reset_index()
county_totals.columns = ['county_fips', 'total_votes']
county_dem = pres_2020[pres_2020['party'] == 'DEMOCRAT'].groupby('county_fips')['candidatevotes'].sum().reset_index()
county_dem.columns = ['county_fips', 'dem_votes']
county_political = county_totals.merge(county_dem, on='county_fips', how='left')
county_political['pct_dem_2020'] = county_political['dem_votes'] / county_political['total_votes']
county_political = county_political[['county_fips', 'pct_dem_2020']]
county_political['county_fips'] = county_political['county_fips'].astype(int)

lea_county = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='5. Counties')
lea_county = lea_county[['1c. LEA ID', '10b. County FIPS Code']].copy()
lea_county.columns = ['nces_id', 'county_fips']
lea_county['nces_id'] = lea_county['nces_id'].astype(str).str.split('.').str[0].str.zfill(7)
lea_county = lea_county.merge(county_political, on='county_fips', how='left')
lea_political = lea_county.groupby('nces_id')['pct_dem_2020'].mean().reset_index()
df = df.merge(lea_political, on='nces_id', how='left')

# Urbanicity dummies
if 'urbanicity' in df.columns:
    urbanicity_dummies = pd.get_dummies(df['urbanicity'], prefix='locale', drop_first=True)
    df = pd.concat([df, urbanicity_dummies], axis=1)
    locale_vars = [col for col in df.columns if col.startswith('locale_')]
else:
    locale_vars = []

# Variables
y_name = 'IS_ADOPTER'
x_names = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pm25', 
           'pct_white', 'pct_dem_2020', 'is_priority'] + locale_vars

# Filter to complete cases (including orthogonalized climate)
climate_vars = ['precip_state_resid', 'tmin_state_resid']
reg_df = df.dropna(subset=[y_name] + x_names + ['nces_id', 'state', 'IV_Z'] + climate_vars).copy()
print(f"Regression sample (with ortho climate): {len(reg_df)}")

# ==============================================================================
# BUILD SPATIAL NETWORKS
# ==============================================================================

print("\n" + "=" * 80)
print("BUILDING SPATIAL NETWORKS")
print("=" * 80)

# Load shapefile and project
print("\nLoading shapefile...")
gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
shp_id_col = 'GEOID' if 'GEOID' in gdf.columns else [c for c in gdf.columns if 'ID' in c][0]
gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.zfill(7)
gdf = gdf.drop_duplicates(subset=shp_id_col)  # Ensure no duplicate geometries
reg_df['nces_id'] = reg_df['nces_id'].astype(str).str.zfill(7)
reg_df = reg_df.drop_duplicates(subset='nces_id')  # Ensure no duplicates

# Merge to reduce before CRS transformation
gdf = gdf.merge(reg_df[['nces_id']], left_on=shp_id_col, right_on='nces_id', how='inner')
print(f"  Matched {len(gdf)} districts")

gdf = gdf.to_crs(epsg=5070)
gdf = gdf.drop(columns=['nces_id'])
geo_df = gdf.merge(reg_df, left_on=shp_id_col, right_on='nces_id', how='inner')
geo_df = geo_df.drop_duplicates(subset='nces_id')  # Final dedup check
geo_df = geo_df.reset_index(drop=True)
geo_df['centroid'] = geo_df.geometry.centroid
geo_df_points = geo_df.set_geometry('centroid')

print(f"Final matched sample: {len(geo_df)}")

# ==============================================================================
# GEOGRAPHIC KNN-6 WEIGHTS
# ==============================================================================

print("\nBuilding geographic weights (KNN-6)...")
w_geo = KNN.from_dataframe(geo_df_points, k=6)
w_geo.transform = 'r'

# Spatial lags for geographic peers
geo_df['w_geo_adoption'] = lag_spatial(w_geo, geo_df['IS_ADOPTER'].values)
geo_df['w_geo_z'] = lag_spatial(w_geo, geo_df['IV_Z'].fillna(0).values)
print(f"  w_geo_adoption range: [{geo_df['w_geo_adoption'].min():.3f}, {geo_df['w_geo_adoption'].max():.3f}]")

# ==============================================================================
# ORTHOGONALIZED CLIMATE KNN-6 WEIGHTS
# ==============================================================================

print("\nBuilding orthogonalized climate weights (KNN-6)...")

# Standardize state-residualized climate
clim_vars = ['precip_state_resid', 'tmin_state_resid']
X_clim = geo_df[clim_vars].values
X_clim_std = (X_clim - X_clim.mean(axis=0)) / X_clim.std(axis=0)

# Use sklearn KNN for memory-efficient neighbor finding
print("  Finding K nearest climate neighbors...")
nn = NearestNeighbors(n_neighbors=7, metric='euclidean')  # 7 to include self
nn.fit(X_clim_std)
distances, indices = nn.kneighbors(X_clim_std)

# Build neighbor dict (skip self which is index 0)
n = len(geo_df)
clim_neighbors = {}
clim_weights = {}

for i in range(n):
    knn_idx = indices[i, 1:]  # Skip self (first neighbor)
    knn_dist = distances[i, 1:]
    # Inverse distance weights, normalized
    w = 1 / (knn_dist + 0.001)
    w = w / w.sum()
    clim_neighbors[i] = list(knn_idx)
    clim_weights[i] = list(w)

# Compute spatial lags using climate neighbors
geo_df['w_clim_ortho_adoption'] = 0.0
geo_df['w_clim_ortho_z'] = 0.0

for i in range(n):
    neighbors = clim_neighbors[i]
    weights = clim_weights[i]
    
    adopt_vals = geo_df.iloc[neighbors]['IS_ADOPTER'].values
    z_vals = geo_df.iloc[neighbors]['IV_Z'].fillna(0).values
    
    geo_df.loc[geo_df.index[i], 'w_clim_ortho_adoption'] = np.dot(weights, adopt_vals)
    geo_df.loc[geo_df.index[i], 'w_clim_ortho_z'] = np.dot(weights, z_vals)

print(f"  w_clim_ortho_adoption range: [{geo_df['w_clim_ortho_adoption'].min():.3f}, {geo_df['w_clim_ortho_adoption'].max():.3f}]")

# ==============================================================================
# DIAGNOSTIC: Overlap between geographic and climate peers
# ==============================================================================

print("\n" + "=" * 80)
print("DIAGNOSTIC: Peer Network Overlap")
print("=" * 80)

# For each district, what fraction of climate peers are also geographic peers?
overlap_fractions = []
for i in range(n):
    geo_neighbors = set(w_geo.neighbors[i])
    clim_neighbors_set = set(clim_neighbors[i])
    overlap = len(geo_neighbors & clim_neighbors_set)
    overlap_fractions.append(overlap / 6)

mean_overlap = np.mean(overlap_fractions)
print(f"\nMean overlap between geo and ortho-climate peers: {mean_overlap:.1%}")
print(f"  If overlap ~100%: Climate peers are just geographic peers")
print(f"  If overlap ~0%: Climate and geographic peers are distinct")

# Correlation between peer lags
corr_adoption = geo_df['w_geo_adoption'].corr(geo_df['w_clim_ortho_adoption'])
corr_z = geo_df['w_geo_z'].corr(geo_df['w_clim_ortho_z'])
print(f"\nCorrelation between peer adoption rates:")
print(f"  Geo vs Ortho-Climate: r = {corr_adoption:.3f}")
print(f"\nCorrelation between peer lottery rates:")
print(f"  Geo vs Ortho-Climate: r = {corr_z:.3f}")

# ==============================================================================
# IV-2SLS: HORSE RACE
# ==============================================================================

print("\n" + "=" * 80)
print("IV-2SLS: GEOGRAPHIC vs ORTHOGONALIZED CLIMATE")
print("=" * 80)

# State fixed effects
state_dummies = pd.get_dummies(geo_df['state'], prefix='st', drop_first=True)
geo_df = pd.concat([geo_df.reset_index(drop=True), state_dummies.reset_index(drop=True)], axis=1)
state_fe_cols = list(state_dummies.columns)

# Control variables
control_vars = ['log_median_income', 'poverty_rate', 'log_enrollment', 
                'pm25', 'pct_white', 'pct_dem_2020', 'is_priority'] + locale_vars

# Prepare matrices
y = geo_df['IS_ADOPTER']

# Model 1: Geographic only
print("\n--- Model 1: Geographic Peers Only ---")
X_endog_1 = geo_df[['w_geo_adoption']]
X_exog_1 = sm.add_constant(geo_df[control_vars + state_fe_cols])
Z_1 = geo_df[['w_geo_z']]

iv_geo = IV2SLS(
    dependent=y,
    exog=X_exog_1,
    endog=X_endog_1,
    instruments=Z_1
).fit(cov_type='clustered', clusters=geo_df['state'])

print(f"  w_geo_adoption: {iv_geo.params['w_geo_adoption']:.4f} (SE={iv_geo.std_errors['w_geo_adoption']:.4f}, p={iv_geo.pvalues['w_geo_adoption']:.4f})")

# Model 2: Orthogonalized Climate only
print("\n--- Model 2: Orthogonalized Climate Peers Only ---")
X_endog_2 = geo_df[['w_clim_ortho_adoption']]
X_exog_2 = sm.add_constant(geo_df[control_vars + state_fe_cols])
Z_2 = geo_df[['w_clim_ortho_z']]

iv_clim = IV2SLS(
    dependent=y,
    exog=X_exog_2,
    endog=X_endog_2,
    instruments=Z_2
).fit(cov_type='clustered', clusters=geo_df['state'])

print(f"  w_clim_ortho_adoption: {iv_clim.params['w_clim_ortho_adoption']:.4f} (SE={iv_clim.std_errors['w_clim_ortho_adoption']:.4f}, p={iv_clim.pvalues['w_clim_ortho_adoption']:.4f})")

# Model 3: Horse Race (both)
print("\n--- Model 3: Horse Race (Both) ---")
X_endog_3 = geo_df[['w_geo_adoption', 'w_clim_ortho_adoption']]
X_exog_3 = sm.add_constant(geo_df[control_vars + state_fe_cols])
Z_3 = geo_df[['w_geo_z', 'w_clim_ortho_z']]

iv_both = IV2SLS(
    dependent=y,
    exog=X_exog_3,
    endog=X_endog_3,
    instruments=Z_3
).fit(cov_type='clustered', clusters=geo_df['state'])

print(f"  w_geo_adoption:        {iv_both.params['w_geo_adoption']:.4f} (SE={iv_both.std_errors['w_geo_adoption']:.4f}, p={iv_both.pvalues['w_geo_adoption']:.4f})")
print(f"  w_clim_ortho_adoption: {iv_both.params['w_clim_ortho_adoption']:.4f} (SE={iv_both.std_errors['w_clim_ortho_adoption']:.4f}, p={iv_both.pvalues['w_clim_ortho_adoption']:.4f})")

# ==============================================================================
# FIRST STAGE DIAGNOSTICS
# ==============================================================================

print("\n--- First Stage Diagnostics ---")

# First stage for geographic - ensure float types
fs_vars = geo_df[control_vars + state_fe_cols + ['w_geo_z', 'w_clim_ortho_z']].astype(float)
fs_geo = sm.OLS(geo_df['w_geo_adoption'].astype(float), 
                sm.add_constant(fs_vars)
               ).fit()
f_geo = fs_geo.tvalues['w_geo_z']**2

# First stage for climate
fs_clim = sm.OLS(geo_df['w_clim_ortho_adoption'].astype(float), 
                 sm.add_constant(fs_vars)
                ).fit()
f_clim = fs_clim.tvalues['w_clim_ortho_z']**2

print(f"  F-stat (w_geo_z on w_geo_adoption): {f_geo:.1f}")
print(f"  F-stat (w_clim_ortho_z on w_clim_ortho_adoption): {f_clim:.1f}")

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================

print("\n" + "=" * 80)
print("SUMMARY: DECOMPOSITION RESULTS")
print("=" * 80)

print(f"""
{'Model':<35} {'Geo Effect':<15} {'Clim Effect':<15}
{'-'*65}
Geographic Only                     {iv_geo.params['w_geo_adoption']:.4f}***        --
Ortho-Climate Only                  --               {iv_clim.params['w_clim_ortho_adoption']:.4f}
Horse Race (Both)                   {iv_both.params['w_geo_adoption']:.4f}           {iv_both.params['w_clim_ortho_adoption']:.4f}

Peer Network Overlap: {mean_overlap:.1%}
Correlation (adoption lags): {corr_adoption:.3f}
""")

# ==============================================================================
# INTERPRETATION
# ==============================================================================

print("=" * 80)
print("INTERPRETATION")
print("=" * 80)

geo_sig = iv_both.pvalues['w_geo_adoption'] < 0.05
clim_sig = iv_both.pvalues['w_clim_ortho_adoption'] < 0.05

if geo_sig and not clim_sig:
    print("""
RESULT: Geographic proximity matters, orthogonalized climate does NOT.

This means:
- Peer effects operate through PROXIMITY (information spillovers, demonstration)
- Climate similarity was just proxying for regional factors
- After removing state-level climate correlation, climate adds nothing

Conclusion: The "climate peer effects" in previous specifications were
an artifact of regional confounding, not a distinct mechanism.
""")
elif geo_sig and clim_sig:
    print("""
RESULT: BOTH geographic and orthogonalized climate effects are significant.

This suggests:
- Peer effects operate through multiple channels
- Geographic proximity captures local spillovers
- Climate similarity captures technical similarity concerns
- Districts may learn from climatically-similar peers even if far away

This is the most interesting finding - distinct mechanisms at play.
""")
elif not geo_sig and clim_sig:
    print("""
RESULT: Orthogonalized climate matters, geographic does NOT.

This is surprising and suggests:
- Peer effects are NOT about proximity
- Districts learn from technically-similar peers regardless of distance
- May indicate virtual/remote learning channels

Requires careful interpretation.
""")
else:
    print("""
RESULT: Neither geographic nor orthogonalized climate effects are significant.

This could mean:
- Sample size/power issues after climate restriction
- Peer effects are weak overall
- Both measures are too noisy to detect effects

Consider: Full sample geographic estimate was significant (0.19).
""")

# Save results
results_df = pd.DataFrame({
    'Model': ['Geographic Only', 'Ortho-Climate Only', 'Horse Race: Geo', 'Horse Race: Clim'],
    'Coefficient': [iv_geo.params['w_geo_adoption'], iv_clim.params['w_clim_ortho_adoption'],
                    iv_both.params['w_geo_adoption'], iv_both.params['w_clim_ortho_adoption']],
    'SE': [iv_geo.std_errors['w_geo_adoption'], iv_clim.std_errors['w_clim_ortho_adoption'],
           iv_both.std_errors['w_geo_adoption'], iv_both.std_errors['w_clim_ortho_adoption']],
    'P-value': [iv_geo.pvalues['w_geo_adoption'], iv_clim.pvalues['w_clim_ortho_adoption'],
                iv_both.pvalues['w_geo_adoption'], iv_both.pvalues['w_clim_ortho_adoption']]
})
results_df.to_csv(TABLES_DIR / "climate_decomposition_results.csv", index=False)
print(f"\nResults saved to: {TABLES_DIR / 'climate_decomposition_results.csv'}")

print("\nDone.")
