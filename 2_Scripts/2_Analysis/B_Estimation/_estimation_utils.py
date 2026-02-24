"""
Shared Estimation Utilities
============================
Single source of truth for data loading, weight construction,
and IV-2SLS estimation across all analysis scripts.

All estimation scripts should import from here to ensure
consistent controls, samples, and methodology.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import libpysal
from libpysal.weights import KNN, W
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
import pickle
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    ESB_FULL_ANALYSIS, SCHOOL_DISTRICTS_SHP, POLITICAL_COUNTY_PRES_FILE,
    WRI_EXCEL_FILE, CLIMATE_SIMILARITY_DATA, TABLES_DIR,
    CLIMATE_KNN_VALUES, get_climate_weight_file, ensure_dirs_exist
)

# =============================================================================
# Canonical Control Variable Sets
# =============================================================================

# Full controls (default for all canonical specifications)
# Note: pct_hispanic/pct_black excluded to avoid multicollinearity with pct_white
FULL_CONTROLS = [
    'log_median_income', 'poverty_rate', 'log_enrollment',
    'pm25', 'pct_white', 'pct_dem_2020', 'is_priority'
]

# Simple controls (only for replicating Script 04 in decomposition)
SIMPLE_CONTROLS = [
    'log_median_income', 'log_enrollment', 'poverty_rate',
    'pct_hispanic', 'pct_white', 'pct_black', 'pm25'
]


# =============================================================================
# Data Loading
# =============================================================================

def load_analysis_data():
    """
    Load the main analysis dataset with all covariates.

    Replicates Script 01 data loading: loads esb_full_analysis_dataset.csv,
    merges pct_dem_2020 from political data via LEA-county mapping,
    creates log transforms and urbanicity dummies.

    Returns
    -------
    df : pd.DataFrame
        Analysis dataset with all variables.
    locale_vars : list
        Names of urbanicity dummy columns.
    """
    print("Loading analysis data...")
    df = pd.read_csv(str(ESB_FULL_ANALYSIS), dtype={'nces_id': str})
    df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)

    # Log transforms
    df['log_median_income'] = np.log(df['median_income'] + 1)
    df['log_enrollment'] = np.log(df['enrollment'] + 1)

    # --- Merge political data (pct_dem_2020) ---
    print("  Merging political data...")
    pres_df = pd.read_csv(str(POLITICAL_COUNTY_PRES_FILE))
    pres_2020 = pres_df[
        (pres_df['year'] == 2020) & (pres_df['office'] == 'US PRESIDENT')
    ].copy()

    county_totals = pres_2020.groupby('county_fips')['candidatevotes'].sum().reset_index()
    county_totals.columns = ['county_fips', 'total_votes']

    county_dem = pres_2020[pres_2020['party'] == 'DEMOCRAT'].groupby(
        'county_fips'
    )['candidatevotes'].sum().reset_index()
    county_dem.columns = ['county_fips', 'dem_votes']

    county_political = county_totals.merge(county_dem, on='county_fips', how='left')
    county_political['pct_dem_2020'] = (
        county_political['dem_votes'] / county_political['total_votes']
    )
    county_political = county_political[['county_fips', 'pct_dem_2020']]
    county_political['county_fips'] = county_political['county_fips'].astype(int)

    # LEA-to-county mapping from WRI data
    lea_county = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='5. Counties')
    lea_county = lea_county[['1c. LEA ID', '10b. County FIPS Code']].copy()
    lea_county.columns = ['nces_id', 'county_fips']
    lea_county['nces_id'] = lea_county['nces_id'].astype(str).str.split('.').str[0].str.zfill(7)
    lea_county = lea_county.merge(county_political, on='county_fips', how='left')

    # For LEAs spanning multiple counties, take mean
    lea_political = lea_county.groupby('nces_id')['pct_dem_2020'].mean().reset_index()

    df = df.merge(lea_political, on='nces_id', how='left')
    n_political = df['pct_dem_2020'].notna().sum()
    print(f"  Political data merged: {n_political}/{len(df)} districts")

    # --- Urbanicity dummies ---
    locale_vars = []
    if 'urbanicity' in df.columns:
        urbanicity_dummies = pd.get_dummies(df['urbanicity'], prefix='locale', drop_first=True)
        df = pd.concat([df, urbanicity_dummies], axis=1)
        locale_vars = [col for col in df.columns if col.startswith('locale_')]
        print(f"  Urbanicity dummies: {locale_vars}")
    else:
        print("  WARNING: urbanicity column not found")

    print(f"  Final dataset: {len(df)} districts")
    return df, locale_vars


def load_shapefile_and_merge(df):
    """
    Load school district shapefile, merge with analysis data,
    project to EPSG:5070, compute centroids.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis dataset (must have 'nces_id' column).

    Returns
    -------
    geo_df : gpd.GeoDataFrame
        Merged spatial+attribute data with centroid geometry.
    """
    print("Loading and merging shapefile...")
    gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))

    # Standardize ID column
    if 'GEOID' in gdf.columns:
        shp_id_col = 'GEOID'
    else:
        possible_cols = [c for c in gdf.columns if 'ID' in c]
        shp_id_col = possible_cols[0]

    gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.zfill(7)
    df['nces_id'] = df['nces_id'].astype(str).str.zfill(7)

    # Memory fix: merge first to reduce before CRS transformation
    gdf = gdf.merge(df[['nces_id']], left_on=shp_id_col, right_on='nces_id', how='inner')
    print(f"  Matched {len(gdf)} districts to shapefile")

    # Project to Albers Equal Area
    gdf = gdf.to_crs(epsg=5070)

    # Merge full covariates
    gdf = gdf.drop(columns=['nces_id'])
    geo_df = gdf.merge(df, left_on=shp_id_col, right_on='nces_id', how='inner')
    geo_df = geo_df.reset_index(drop=True)

    # Centroids for KNN
    geo_df['centroid'] = geo_df.geometry.centroid

    print(f"  Final spatial sample: {len(geo_df)} districts")
    return geo_df


# =============================================================================
# Weight Matrix Construction
# =============================================================================

def build_geo_knn(geo_df, k):
    """
    Build geographic KNN weight matrix from centroids.

    Parameters
    ----------
    geo_df : gpd.GeoDataFrame
        Must have 'centroid' column.
    k : int
        Number of nearest neighbors.

    Returns
    -------
    w : libpysal.weights.W
        Row-standardized KNN weight matrix.
    """
    geo_points = geo_df.set_geometry('centroid')
    w = KNN.from_dataframe(geo_points, k=k)
    w.transform = 'r'
    return w


def build_geo_donut(geo_df, k_inner, k_outer):
    """
    Build geographic "donut" weight matrix: KNN-k_outer minus KNN-k_inner.
    Keeps only neighbors ranked (k_inner+1) through k_outer.

    This is the approach used in Script 01 (KNN-6 inner, KNN-20 outer).

    Parameters
    ----------
    geo_df : gpd.GeoDataFrame
        Must have 'centroid' column.
    k_inner : int
        Inner ring cutoff (neighbors 1..k_inner excluded).
    k_outer : int
        Outer ring cutoff (neighbors k_inner+1..k_outer included).

    Returns
    -------
    w : libpysal.weights.W
        Row-standardized donut weight matrix.
    """
    geo_points = geo_df.set_geometry('centroid')
    w_base = KNN.from_dataframe(geo_points, k=k_outer)

    neighbors_donut = {}
    weights_donut = {}

    for idx in range(len(geo_df)):
        all_neighbors = w_base.neighbors[idx]
        donut_peers = all_neighbors[k_inner:]
        neighbors_donut[idx] = donut_peers
        weights_donut[idx] = [1] * len(donut_peers)

    w = W(neighbors_donut, weights_donut)
    w.transform = 'r'
    return w


def load_and_align_climate_weights(geo_df, k):
    """
    Load climate-based KNN weights from pickle and align to geo_df's sample.

    Consolidates the duplicated alignment logic from Scripts 04 and 05.
    Subsets geo_df to districts with climate data and rebuilds geographic
    weights on the subset.

    Parameters
    ----------
    geo_df : gpd.GeoDataFrame
        Full spatial sample. Must have 'nces_id' and 'centroid' columns.
    k : int
        KNN parameter for both climate and geographic weights.

    Returns
    -------
    geo_df_subset : gpd.GeoDataFrame
        Subset with climate data (reset index).
    w_geo_subset : libpysal.weights.W
        Geographic KNN-k on the subset.
    w_clim_subset : libpysal.weights.W
        Climate KNN-k aligned to the subset.
    """
    # Load climate pickle
    climate_file = get_climate_weight_file(k)
    print(f"  Loading climate weights from {climate_file.name}...")
    with open(climate_file, 'rb') as f:
        w_clim_full = pickle.load(f)

    # Load climate normals to get ncessch-to-index mapping
    df_clim = pd.read_csv(str(CLIMATE_SIMILARITY_DATA), dtype={'ncessch': str})

    # Standardize IDs
    geo_ncessch = geo_df['nces_id'].astype(str).str.zfill(7).values

    # Build lookups: ncessch -> climate index -> clean sample index
    ncessch_to_clim_idx = {
        ncessch: idx for idx, ncessch in enumerate(df_clim['ncessch'].values)
    }

    clim_idx_to_clean_idx = {}
    for clean_idx, ncessch in enumerate(geo_ncessch):
        if ncessch in ncessch_to_clim_idx:
            clim_idx = ncessch_to_clim_idx[ncessch]
            clim_idx_to_clean_idx[clim_idx] = clean_idx

    print(f"  Matched {len(clim_idx_to_clean_idx)} districts to climate data")

    # Subset to districts with climate data
    districts_with_climate = sorted(set(clim_idx_to_clean_idx.values()))
    geo_df_subset = geo_df.iloc[districts_with_climate].reset_index(drop=True)

    print(f"  Climate-restricted sample: {len(geo_df_subset)} districts")

    # Rebuild geographic weights on subset
    w_geo_subset = build_geo_knn(geo_df_subset, k)

    # Remap climate weights to new indices
    old_to_new_idx = {
        old_idx: new_idx for new_idx, old_idx in enumerate(districts_with_climate)
    }

    neighbors_subset = {}
    weights_subset = {}

    for clim_idx in sorted(clim_idx_to_clean_idx.keys()):
        old_clean_idx = clim_idx_to_clean_idx[clim_idx]
        new_clean_idx = old_to_new_idx[old_clean_idx]

        clim_neighbors = w_clim_full.neighbors[clim_idx]
        clim_weights_orig = w_clim_full.weights[clim_idx]

        new_neighbors = []
        new_weights = []

        for j, clim_j in enumerate(clim_neighbors):
            if clim_j in clim_idx_to_clean_idx:
                old_neighbor_idx = clim_idx_to_clean_idx[clim_j]
                new_neighbor_idx = old_to_new_idx[old_neighbor_idx]
                new_neighbors.append(new_neighbor_idx)
                new_weights.append(clim_weights_orig[j])

        if new_neighbors:
            neighbors_subset[new_clean_idx] = new_neighbors
            weights_subset[new_clean_idx] = new_weights

    w_clim_subset = W(neighbors_subset, weights_subset)
    w_clim_subset.transform = 'r'

    # Validation check
    assert w_clim_subset.n == len(geo_df_subset), (
        f"Climate weight N ({w_clim_subset.n}) != subset N ({len(geo_df_subset)})"
    )

    print(f"  Climate weights aligned: {w_clim_subset.n} districts, "
          f"mean neighbors: {w_clim_subset.mean_neighbors:.1f}")

    return geo_df_subset, w_geo_subset, w_clim_subset


# =============================================================================
# Spatial Lag Construction
# =============================================================================

def create_spatial_lags(geo_df, w_geo, w_clim=None, y_name='IS_ADOPTER', z_name='IV_Z'):
    """
    Create spatial lags of adoption and instruments.

    Parameters
    ----------
    geo_df : gpd.GeoDataFrame
        Must contain y_name and z_name columns.
    w_geo : libpysal.weights.W
        Geographic weight matrix.
    w_clim : libpysal.weights.W or None
        Climate weight matrix (optional).
    y_name : str
        Dependent variable name.
    z_name : str
        Instrument variable name.

    Returns
    -------
    geo_df : gpd.GeoDataFrame
        With added spatial lag columns.
    """
    geo_df = geo_df.copy()
    geo_df['w_geo_adoption'] = libpysal.weights.lag_spatial(w_geo, geo_df[y_name].values)
    geo_df['w_geo_z'] = libpysal.weights.lag_spatial(w_geo, geo_df[z_name].values)

    if w_clim is not None:
        geo_df['w_clim_adoption'] = libpysal.weights.lag_spatial(w_clim, geo_df[y_name].values)
        geo_df['w_clim_z'] = libpysal.weights.lag_spatial(w_clim, geo_df[z_name].values)

    return geo_df


# =============================================================================
# IV-2SLS Estimation
# =============================================================================

def run_iv2sls(df, y_name='IS_ADOPTER', endog_names=None, instrument_names=None,
               control_names=None, locale_vars=None, state_fe=True,
               cluster_var='state', include_own_z=True):
    """
    Run canonical 2SLS estimation with named DataFrames.

    Coefficients are extracted by name (not positional indexing) to avoid
    fragile iloc-based extraction.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with all needed variables.
    y_name : str
        Dependent variable.
    endog_names : list of str
        Endogenous variable names (e.g., ['w_geo_adoption', 'w_clim_adoption']).
    instrument_names : list of str
        Instrument names (e.g., ['w_geo_z', 'w_clim_z']).
    control_names : list of str
        Control variable names.
    locale_vars : list of str or None
        Urbanicity dummy column names.
    state_fe : bool
        Whether to include state fixed effects.
    cluster_var : str or None
        Variable name for clustering SEs. None = default (heteroskedastic-robust).
    include_own_z : bool
        Whether to include IV_Z as an exogenous control.

    Returns
    -------
    dict with keys: params, std_errors, pvalues, f_stats, n, r_squared,
                    r_squared_adj, model (the fitted model object).
    """
    if endog_names is None:
        endog_names = ['w_geo_adoption']
    if instrument_names is None:
        instrument_names = ['w_geo_z']
    if control_names is None:
        control_names = FULL_CONTROLS
    if locale_vars is None:
        locale_vars = []

    # Build exogenous variable set
    exog_vars = list(control_names) + list(locale_vars)
    if include_own_z and 'IV_Z' in df.columns and 'IV_Z' not in exog_vars:
        exog_vars.append('IV_Z')

    # Drop rows with NaN in any required variable
    all_needed = [y_name] + exog_vars + endog_names + instrument_names
    if cluster_var:
        all_needed.append(cluster_var)
    if state_fe:
        all_needed.append('state')

    df_reg = df.dropna(subset=list(set(all_needed))).copy()

    # Build named DataFrames
    dependent = df_reg[y_name]
    endog = df_reg[endog_names]
    instruments = df_reg[instrument_names]

    # Exogenous: controls + optional state FE
    exog_parts = [df_reg[exog_vars]]

    if state_fe:
        state_dummies = pd.get_dummies(df_reg['state'], prefix='state', drop_first=True)
        exog_parts.append(state_dummies)

    exog = sm.add_constant(pd.concat(exog_parts, axis=1))

    # Fit model
    model = IV2SLS(dependent, exog, endog, instruments)
    if cluster_var and cluster_var in df_reg.columns:
        results = model.fit(cov_type='clustered', clusters=df_reg[cluster_var])
    else:
        results = model.fit(cov_type='robust')

    # Extract results by name
    result_dict = {
        'n': len(df_reg),
        'r_squared': results.rsquared,
        'r_squared_adj': results.rsquared_adj,
        'params': {},
        'std_errors': {},
        'pvalues': {},
        'f_stats': {},
    }

    # Extract coefficients for endogenous variables
    for var in endog_names:
        result_dict['params'][var] = results.params[var]
        result_dict['std_errors'][var] = results.std_errors[var]
        result_dict['pvalues'][var] = results.pvalues[var]

    # Extract coefficients for control variables
    for var in exog_vars:
        if var in results.params.index:
            result_dict['params'][var] = results.params[var]
            result_dict['std_errors'][var] = results.std_errors[var]
            result_dict['pvalues'][var] = results.pvalues[var]

    # First-stage F-statistics
    try:
        for var in endog_names:
            result_dict['f_stats'][var] = results.first_stage.diagnostics['f.stat'][var]
    except Exception:
        # Fallback: try positional access
        try:
            for i, var in enumerate(endog_names):
                result_dict['f_stats'][var] = results.first_stage.diagnostics.iloc[i]['f.stat']
        except Exception:
            for var in endog_names:
                result_dict['f_stats'][var] = np.nan

    # State count
    if 'state' in df_reg.columns:
        result_dict['n_states'] = df_reg['state'].nunique()

    # Adopter count
    result_dict['n_adopters'] = int(df_reg[y_name].sum())
    result_dict['adoption_rate'] = df_reg[y_name].mean()

    # Store fitted model for further inspection
    result_dict['model'] = results

    return result_dict


# =============================================================================
# Results Formatting
# =============================================================================

def format_result_row(spec_label, sample_label, result, endog_names=None):
    """
    Format a single estimation result into a flat dictionary for CSV output.

    Parameters
    ----------
    spec_label : str
        Specification name (e.g., 'KNN-6', 'S1').
    sample_label : str
        Sample name (e.g., 'Full', 'Climate-Restricted').
    result : dict
        Output from run_iv2sls().
    endog_names : list or None
        Names of endogenous variables to include.

    Returns
    -------
    dict : Flat dictionary suitable for pd.DataFrame row.
    """
    if endog_names is None:
        endog_names = list(result['params'].keys())

    row = {
        'specification': spec_label,
        'sample': sample_label,
        'n_obs': result['n'],
        'n_states': result.get('n_states', np.nan),
        'n_adopters': result.get('n_adopters', np.nan),
        'adoption_rate': result.get('adoption_rate', np.nan),
        'r_squared': result['r_squared'],
    }

    # Add peer effect coefficients
    if 'w_geo_adoption' in result['params']:
        row['w_geo_coef'] = result['params']['w_geo_adoption']
        row['w_geo_se'] = result['std_errors']['w_geo_adoption']
        row['w_geo_pval'] = result['pvalues']['w_geo_adoption']
        row['f_stat_geo'] = result['f_stats'].get('w_geo_adoption', np.nan)

    if 'w_clim_adoption' in result['params']:
        row['w_clim_coef'] = result['params']['w_clim_adoption']
        row['w_clim_se'] = result['std_errors']['w_clim_adoption']
        row['w_clim_pval'] = result['pvalues']['w_clim_adoption']
        row['f_stat_clim'] = result['f_stats'].get('w_clim_adoption', np.nan)

    return row


def significance_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.10:
        return '*'
    return ''


def save_results_csv(results_list, output_path):
    """Save a list of result row dicts to CSV."""
    df = pd.DataFrame(results_list)
    df.to_csv(str(output_path), index=False)
    print(f"Results saved to {output_path}")
    return df


def print_results_table(results_list, title="Estimation Results"):
    """Print a formatted results table to console."""
    df = pd.DataFrame(results_list)

    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")

    # Select display columns based on what's available
    display_cols = ['specification', 'sample', 'n_obs']
    if 'w_geo_coef' in df.columns:
        display_cols += ['w_geo_coef', 'w_geo_se', 'w_geo_pval', 'f_stat_geo']
    if 'w_clim_coef' in df.columns:
        display_cols += ['w_clim_coef', 'w_clim_se', 'w_clim_pval', 'f_stat_clim']

    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False))
    print()
