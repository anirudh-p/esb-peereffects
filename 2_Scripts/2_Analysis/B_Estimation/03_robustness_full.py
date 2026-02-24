import pandas as pd
import geopandas as gpd
import numpy as np
import libpysal 
from libpysal.weights import KNN, DistanceBand
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, WRI_EXCEL_FILE,
    POLITICAL_COUNTY_PRES_FILE, CHARGING_STATIONS_FILE, ZIP_COUNTY_FILE,
    TABLES_DIR, ensure_dirs_exist
)

# Ensure output directories exist
ensure_dirs_exist()

# --- CONFIGURATION ---
OUTPUT_RESULTS_FILE = TABLES_DIR / 'esb_spatial_results_robustness.txt'
OUTPUT_CSV = TABLES_DIR / 'esb_robustness_results.csv'

print("="*80)
print("ESB ADOPTION PEER EFFECTS: COMPREHENSIVE ROBUSTNESS ANALYSIS")
print("="*80)

# --- 1. DATA PREP ---
print("\n[1/7] Loading and Preparing Base Data...")
df = pd.read_csv(str(ESB_FULL_ANALYSIS))  # nces_id already stored as int64 in CSV

# Charter flag already in dataset
print(f"  Charter school districts in sample: {df['is_charter'].sum()}")

# Engineered features
df['log_median_income'] = np.log(df['median_income'] + 1)
df['log_enrollment'] = np.log(df['enrollment'] + 1)

# --- 2. LOAD UTILITY DATA ---
print("\n[2/7] Loading Utility Ownership Data...")
util_df = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='4. Utilities')

# Clean LEA ID - filter non-numeric and convert to int
util_df['lea_id_str'] = util_df['1c. LEA ID'].astype(str).str.replace('.0', '', regex=False)
util_df = util_df[util_df['lea_id_str'].str.isdigit()].copy()
util_df['nces_id'] = util_df['lea_id_str'].astype(int)

# Extract utility ownership type (dominant category per LEA)
ownership_cols = {
    'coop': '9b. Cooperative ownership',
    'federal': '9c. Federal ownership',
    'investor': '9d. Investor ownership',
    'municipal': '9e. Municipal ownership',
    'muni_marketing': '9f. Municipal marketing authority',
    'political_subdiv': '9g. Political subdivision',
    'state': '9h. State ownership',
    'wholesale': '9i. Wholesale'
}

# Create utility type variable (most common ownership across utilities serving LEA)
for short, col in ownership_cols.items():
    util_df[short] = util_df[col].fillna(0)

util_df['utility_type'] = util_df[[col for col in ownership_cols.keys()]].idxmax(axis=1)
util_df['utility_type'] = util_df['utility_type'].fillna('unknown')

util_agg = util_df.groupby('nces_id')['utility_type'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown').reset_index()

print(f"  Loaded utility data for {len(util_agg)} districts")
print(f"  Utility types: {util_agg['utility_type'].value_counts().to_dict()}")

# Merge utility data
df = df.merge(util_agg, on='nces_id', how='left')
df['utility_type'] = df['utility_type'].fillna('unknown')

# --- 3. LOAD POLITICAL DATA (COUNTY-LEVEL) ---
print("\n[3/7] Loading County-Level Political Data...")
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

# Load LEA-to-county mapping
lea_county = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='5. Counties')
lea_county = lea_county[['1c. LEA ID', '10b. County FIPS Code']].copy()
lea_county.columns = ['nces_id', 'county_fips']
lea_county['nces_id'] = lea_county['nces_id'].astype(str).str.split('.').str[0].str.zfill(7)

lea_county = lea_county.merge(county_political, on='county_fips', how='left')
lea_political = lea_county.groupby('nces_id')['pct_dem_2020'].mean().reset_index()

df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)
df = df.merge(lea_political, on='nces_id', how='left')
print(f"  Merged political data: {df['pct_dem_2020'].notna().sum()} / {len(df)} districts")

# --- 4. LOAD AND PROCESS EV CHARGING INFRASTRUCTURE ---
print("\n[4/7] Loading EV Charging Station Data...")
charging_df = pd.read_csv(str(CHARGING_STATIONS_FILE), low_memory=False)

# Filter to electric charging stations
charging_ev = charging_df[charging_df['Fuel Type Code'] == 'ELEC'].copy()
print(f"  Total EV charging stations: {len(charging_ev)}")

# Extract relevant columns
charging_ev['zip'] = charging_ev['ZIP'].astype(str).str[:5]  # First 5 digits
charging_ev['level1'] = pd.to_numeric(charging_ev['EV Level1 EVSE Num'], errors='coerce').fillna(0)
charging_ev['level2'] = pd.to_numeric(charging_ev['EV Level2 EVSE Num'], errors='coerce').fillna(0)
charging_ev['dc_fast'] = pd.to_numeric(charging_ev['EV DC Fast Count'], errors='coerce').fillna(0)

# Filter to valid geocodes
charging_ev = charging_ev[charging_ev['Geocode Status'].isin(['GPS', '200-9'])].copy()

# Option to filter by Open Date (currently commented out as per request)
# charging_ev['open_date'] = pd.to_datetime(charging_ev['Open Date'], errors='coerce')
# charging_ev = charging_ev[(charging_ev['open_date'].isna()) | (charging_ev['open_date'] < '2020-01-01')]

# Aggregate to ZIP level
zip_charging = charging_ev.groupby('zip').agg({
    'level1': 'sum',
    'level2': 'sum',
    'dc_fast': 'sum'
}).reset_index()
zip_charging['total_chargers'] = zip_charging['level1'] + zip_charging['level2'] + zip_charging['dc_fast']

# Filter to valid US ZIP codes (numeric only, 5 digits)
zip_charging = zip_charging[zip_charging['zip'].str.isdigit()].copy()
zip_charging['zip'] = zip_charging['zip'].astype(int)

print(f"  Aggregated to {len(zip_charging)} ZIP codes")

# Load ZIP-County crosswalk
zip_county = pd.read_excel(ZIP_COUNTY_FILE)
zip_county = zip_county[['ZIP', 'COUNTY', 'TOT_RATIO']].copy()
zip_county.columns = ['zip', 'county_fips', 'allocation_ratio']

# Merge charging data to ZIP-County crosswalk
zip_county = zip_county.merge(zip_charging, on='zip', how='left')
zip_county[['level1', 'level2', 'dc_fast', 'total_chargers']] = zip_county[['level1', 'level2', 'dc_fast', 'total_chargers']].fillna(0)

# Allocate charging stations to counties (weighted by allocation ratio)
zip_county['level1_allocated'] = zip_county['level1'] * zip_county['allocation_ratio']
zip_county['level2_allocated'] = zip_county['level2'] * zip_county['allocation_ratio']
zip_county['dc_fast_allocated'] = zip_county['dc_fast'] * zip_county['allocation_ratio']
zip_county['total_chargers_allocated'] = zip_county['total_chargers'] * zip_county['allocation_ratio']

# Aggregate to county level
county_charging = zip_county.groupby('county_fips').agg({
    'level1_allocated': 'sum',
    'level2_allocated': 'sum',
    'dc_fast_allocated': 'sum',
    'total_chargers_allocated': 'sum'
}).reset_index()

county_charging.columns = ['county_fips', 'charging_level1', 'charging_level2', 'charging_dcfast', 'charging_total']
print(f"  Allocated to {len(county_charging)} counties")

# Merge to LEA-County mapping, then aggregate to LEA level
lea_county_full = lea_county.merge(county_charging, on='county_fips', how='left')
lea_charging = lea_county_full.groupby('nces_id').agg({
    'charging_level1': 'mean',
    'charging_level2': 'mean',
    'charging_dcfast': 'mean',
    'charging_total': 'mean'
}).reset_index()

lea_charging[['charging_level1', 'charging_level2', 'charging_dcfast', 'charging_total']] = lea_charging[['charging_level1', 'charging_level2', 'charging_dcfast', 'charging_total']].fillna(0)

# Merge to main dataset
df = df.merge(lea_charging, on='nces_id', how='left')
df[['charging_level1', 'charging_level2', 'charging_dcfast', 'charging_total']] = df[['charging_level1', 'charging_level2', 'charging_dcfast', 'charging_total']].fillna(0)

print(f"  Merged charging data: {(df['charging_total'] > 0).sum()} / {len(df)} districts have nearby chargers")
print(f"  Mean chargers per district: {df['charging_total'].mean():.2f}")

# --- 5. CREATE URBANICITY AND CONTROL VARIABLES ---
print("\n[5/7] Creating Control Variables...")
if 'urbanicity' in df.columns:
    urbanicity_dummies = pd.get_dummies(df['urbanicity'], prefix='locale', drop_first=True)
    df = pd.concat([df, urbanicity_dummies], axis=1)
    locale_vars = [col for col in df.columns if col.startswith('locale_')]
else:
    locale_vars = []

# Create utility type dummies
utility_dummies = pd.get_dummies(df['utility_type'], prefix='utility', drop_first=True)
df = pd.concat([df, utility_dummies], axis=1)
utility_vars = [col for col in df.columns if col.startswith('utility_')]

print(f"  Urbanicity categories: {df['urbanicity'].unique() if 'urbanicity' in df.columns else 'N/A'}")
print(f"  Utility types: {df['utility_type'].value_counts().to_dict()}")

# Base control variables (excluding utility for robustness simplicity - focus on peer definitions)
base_controls = ['log_median_income', 'poverty_rate', 'log_enrollment', 'pm25', 'pct_white', 
                 'pct_dem_2020', 'charging_total', 'is_priority']

y_name = 'IS_ADOPTER'
x_names_A = base_controls + locale_vars
x_names_B = base_controls + locale_vars + ['IV_Z']

# Clean data
all_vars_needed = list(set(x_names_A + x_names_B))
reg_df = df.dropna(subset=[y_name] + all_vars_needed + ['nces_id', 'state', 'is_charter']).copy()
print(f"  Regression Sample Size: {len(reg_df)}")
print(f"  Charter schools: {reg_df['is_charter'].sum()} ({100*reg_df['is_charter'].mean():.1f}%)")
print(f"  Non-charter districts: {(reg_df['is_charter']==0).sum()}")

# Ensure nces_id is integer for spatial merge
reg_df['nces_id'] = reg_df['nces_id'].astype(int)

# --- 6. SPATIAL MATRIX CONSTRUCTION ---
print("\n[6/7] Constructing Spatial Weight Matrices...")
gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
if 'GEOID' not in gdf.columns:
    shp_id_col = [c for c in gdf.columns if 'ID' in c][0]
else:
    shp_id_col = 'GEOID'

# Convert shapefile IDs to integer (they may be stored as strings with leading zeros)
gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.lstrip('0').astype(int)

# Merge and project
gdf_reg = gdf[[shp_id_col, 'geometry']].merge(reg_df, left_on=shp_id_col, right_on='nces_id', how='inner')
gdf_reg = gdf_reg.to_crs(epsg=5070)  # Albers Equal Area
gdf_reg['centroid'] = gdf_reg.geometry.centroid

print(f"  Matched {len(gdf_reg)} districts to shapefile")
print(f"  Districts with valid geometries: {len(gdf_reg)}")

# Create geo_df for analysis - use centroids for ALL districts (including charters)
# This ensures consistent distance calculations
geo_df = gdf_reg.copy()
geo_df = geo_df.set_geometry('centroid')

print(f"  Final sample for spatial analysis: {len(geo_df)} districts")
print(f"    - Charter schools: {geo_df['is_charter'].sum()}")
print(f"    - Non-charter districts: {(geo_df['is_charter']==0).sum()}")

# --- 7. ROBUSTNESS ANALYSIS: MULTIPLE PEER DEFINITIONS ---
print("\n[7/7] Running Robustness Analysis Across Peer Definitions...")
print("="*80)

# Define peer specifications
knn_specs = [4, 6, 8, 10, 12, 15, 20]
distance_specs = [25000, 50000, 75000, 100000]  # meters (25km, 50km, 75km, 100km)

results_list = []

def run_2sls_spec(geo_df_subset, w_geo, w_clim, spec_name, sample_name):
    """Run 2SLS for a given spatial weight specification"""
    
    # Calculate spatial lags
    geo_df_subset = geo_df_subset.copy()
    geo_df_subset['w_geo_adoption'] = libpysal.weights.lag_spatial(w_geo, geo_df_subset[y_name].values)
    geo_df_subset['w_clim_adoption'] = libpysal.weights.lag_spatial(w_clim, geo_df_subset[y_name].values)
    geo_df_subset['w_geo_z'] = libpysal.weights.lag_spatial(w_geo, geo_df_subset['IV_Z'].values)
    geo_df_subset['w_clim_z'] = libpysal.weights.lag_spatial(w_clim, geo_df_subset['IV_Z'].values)
    
    # Prepare data
    dependent = geo_df_subset[y_name]
    endog = geo_df_subset[['w_geo_adoption', 'w_clim_adoption']]
    instruments = geo_df_subset[['w_geo_z', 'w_clim_z']]
    clusters = geo_df_subset['state']
    
    # Version B: With own lottery control (NO STATE FE per user request)
    exog_B = geo_df_subset[x_names_B].copy()
    exog_B = sm.add_constant(exog_B)
    
    try:
        model_iv = IV2SLS(dependent, exog_B, endog, instruments).fit(cov_type='clustered', clusters=clusters)
        
        # Extract key results
        result = {
            'specification': spec_name,
            'sample': sample_name,
            'n_obs': len(geo_df_subset),
            'w_geo_coef': model_iv.params['w_geo_adoption'],
            'w_geo_se': model_iv.std_errors['w_geo_adoption'],
            'w_geo_pval': model_iv.pvalues['w_geo_adoption'],
            'w_clim_coef': model_iv.params['w_clim_adoption'],
            'w_clim_se': model_iv.std_errors['w_clim_adoption'],
            'w_clim_pval': model_iv.pvalues['w_clim_adoption'],
            'iv_z_coef': model_iv.params['IV_Z'],
            'iv_z_se': model_iv.std_errors['IV_Z'],
            'r_squared': model_iv.rsquared,
            'f_stat_geo': model_iv.first_stage.diagnostics['f.stat']['w_geo_adoption'],
            'f_stat_clim': model_iv.first_stage.diagnostics['f.stat']['w_clim_adoption']
        }
        return result
    except Exception as e:
        print(f"    WARNING: {spec_name} ({sample_name}) failed: {str(e)}")
        return None

# --- KNN SPECIFICATIONS ---
print("\nA. K-NEAREST NEIGHBORS SPECIFICATIONS")
print("-" * 80)

for k in knn_specs:
    print(f"\n  KNN-{k}:")
    
    # Build weights on FULL sample (including charters)
    # KNN can handle point geometries (centroids) for all districts
    w_geo = KNN.from_dataframe(geo_df, k=k)
    w_geo.transform = 'r'  # Row-standardize
    
    # For climate (regional), use KNN-(k+10) minus KNN-k
    k_clim = min(k + 10, 25)
    w_full = KNN.from_dataframe(geo_df, k=k_clim)
    w_full.transform = 'r'
    
    # Subtract KNN-k from KNN-(k+10) to get donut
    w_clim_dict = {}
    for i in w_full.neighbors.keys():
        full_neighbors = set(w_full.neighbors[i])
        geo_neighbors = set(w_geo.neighbors[i])
        clim_neighbors = list(full_neighbors - geo_neighbors)
        if clim_neighbors:
            w_clim_dict[i] = clim_neighbors
    
    from libpysal.weights import W
    w_clim = W(w_clim_dict)
    w_clim.transform = 'r'
    
    # Full sample (including charters)
    result_full = run_2sls_spec(geo_df, w_geo, w_clim, f'KNN-{k}', 'Full Sample')
    if result_full:
        results_list.append(result_full)
        print(f"    Full Sample (N={result_full['n_obs']}): w_geo={result_full['w_geo_coef']:.4f} (p={result_full['w_geo_pval']:.3f}), w_clim={result_full['w_clim_coef']:.4f} (p={result_full['w_clim_pval']:.3f})")
    
    # Without charters - rebuild weights on non-charter subset
    geo_df_no_charter = geo_df[geo_df['is_charter'] == 0].copy().reset_index(drop=True)
    
    # Rebuild KNN weights on the non-charter sample
    w_geo_nc = KNN.from_dataframe(geo_df_no_charter, k=k)
    w_geo_nc.transform = 'r'
    
    k_clim_nc = min(k + 10, 25)
    w_full_nc = KNN.from_dataframe(geo_df_no_charter, k=k_clim_nc)
    w_full_nc.transform = 'r'
    
    w_clim_dict_nc = {}
    for i in w_full_nc.neighbors.keys():
        full_neighbors = set(w_full_nc.neighbors[i])
        geo_neighbors = set(w_geo_nc.neighbors[i])
        clim_neighbors = list(full_neighbors - geo_neighbors)
        if clim_neighbors:
            w_clim_dict_nc[i] = clim_neighbors
    
    w_clim_nc = W(w_clim_dict_nc)
    w_clim_nc.transform = 'r'
    
    result_no_charter = run_2sls_spec(geo_df_no_charter, w_geo_nc, w_clim_nc, f'KNN-{k}', 'No Charters')
    if result_no_charter:
        results_list.append(result_no_charter)
        print(f"    No Charters (N={result_no_charter['n_obs']}): w_geo={result_no_charter['w_geo_coef']:.4f} (p={result_no_charter['w_geo_pval']:.3f}), w_clim={result_no_charter['w_clim_coef']:.4f} (p={result_no_charter['w_clim_pval']:.3f})")

# --- DISTANCE BAND SPECIFICATIONS ---
print("\n\nB. DISTANCE BAND SPECIFICATIONS")
print("-" * 80)

for dist_m in distance_specs:
    dist_km = dist_m / 1000
    print(f"\n  Distance {dist_km:.0f}km:")
    
    # Build distance weights
    w_geo = DistanceBand.from_dataframe(geo_df, threshold=dist_m, binary=True)
    w_geo.transform = 'r'
    
    # For regional, use larger distance
    dist_clim = dist_m * 2
    w_clim = DistanceBand.from_dataframe(geo_df, threshold=dist_clim, binary=True)
    w_clim.transform = 'r'
    
    # Full sample
    result_full = run_2sls_spec(geo_df, w_geo, w_clim, f'Distance-{dist_km:.0f}km', 'Full Sample')
    if result_full:
        results_list.append(result_full)
        print(f"    Full Sample (N={result_full['n_obs']}): w_geo={result_full['w_geo_coef']:.4f} (p={result_full['w_geo_pval']:.3f}), w_clim={result_full['w_clim_coef']:.4f} (p={result_full['w_clim_pval']:.3f})")
    
    # Without charters - rebuild weights on non-charter subset
    geo_df_no_charter = geo_df[geo_df['is_charter'] == 0].copy().reset_index(drop=True)
    
    w_geo_nc = DistanceBand.from_dataframe(geo_df_no_charter, threshold=dist_m, binary=True)
    w_geo_nc.transform = 'r'
    
    w_clim_nc = DistanceBand.from_dataframe(geo_df_no_charter, threshold=dist_clim, binary=True)
    w_clim_nc.transform = 'r'
    
    result_no_charter = run_2sls_spec(geo_df_no_charter, w_geo_nc, w_clim_nc, f'Distance-{dist_km:.0f}km', 'No Charters')
    if result_no_charter:
        results_list.append(result_no_charter)
        print(f"    No Charters (N={result_no_charter['n_obs']}): w_geo={result_no_charter['w_geo_coef']:.4f} (p={result_no_charter['w_geo_pval']:.3f}), w_clim={result_no_charter['w_clim_coef']:.4f} (p={result_no_charter['w_clim_pval']:.3f})")

# --- SAVE RESULTS ---
print("\n\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_df = pd.DataFrame(results_list)
results_df.to_csv(str(OUTPUT_CSV), index=False)
print(f"\nRobustness results saved to: {OUTPUT_CSV}")
print(f"Total specifications run: {len(results_df)}")

# Create summary table
with open(str(OUTPUT_RESULTS_FILE), 'w') as f:
    f.write("="*80 + "\n")
    f.write("ESB ADOPTION PEER EFFECTS: ROBUSTNESS ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Sample Size: {len(reg_df)} districts\n")
    f.write(f"Potential Charter Schools: {reg_df['is_charter_proxy'].sum()}\n")
    f.write(f"Districts with Charging Infrastructure: {(reg_df['charging_total'] > 0).sum()}\n")
    f.write(f"Mean Chargers per District: {reg_df['charging_total'].mean():.2f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("ROBUSTNESS TABLE: PEER EFFECT ESTIMATES ACROSS SPECIFICATIONS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"{'Specification':<25} {'Sample':<15} {'N':>8} {'W_Geo_Coef':>12} {'W_Geo_SE':>10} {'W_Geo_p':>10} {'W_Clim_Coef':>12} {'W_Clim_SE':>10} {'W_Clim_p':>10}\n")
    f.write("-"*130 + "\n")
    
    for _, row in results_df.iterrows():
        f.write(f"{row['specification']:<25} {row['sample']:<15} {row['n_obs']:>8} "
                f"{row['w_geo_coef']:>12.4f} {row['w_geo_se']:>10.4f} {row['w_geo_pval']:>10.4f} "
                f"{row['w_clim_coef']:>12.4f} {row['w_clim_se']:>10.4f} {row['w_clim_pval']:>10.4f}\n")

print(f"\nFull results saved to: {OUTPUT_RESULTS_FILE}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
