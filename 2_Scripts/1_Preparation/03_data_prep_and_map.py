"""
03_data_prep_and_map.py
========================
Load and merge EPA Clean School Bus Program data (rebates, grants, applicants).
Creates master regression dataset and adoption status maps.

Inputs:
    - Raw/Other/CSB_Rebates.xlsx
    - Raw/Other/CSB_Grants.xlsx
    - Raw/Other/CSBP Applicants waitlisted and rejected_11.18.25.xlsx
    - Raw/WRI/ESB_adoption_dataset_v9_update_june_2025.xlsx (for bus-level data)
    - Raw/Spatial/EDGE_SCHOOLDISTRICT_TL21_SY2021/schooldistrict_sy2021_tl21.shp

Outputs:
    - Cleaned/esb_master_data_for_regression.csv
    - Figures/esb_adoption_status_map.png
    - Figures/esb_adoption_intensity_map.png
    - Figures/esb_adoption_by_year_map.png

Dependencies: None (independent preparation step)

Author: [Your Name]
Last Updated: November 30, 2025
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SCHOOL_DISTRICTS_SHP, ADOPTION_REBATES_FILE, ADOPTION_GRANTS_FILE,
    APPLICANT_FILE, WRI_EXCEL_FILE, ESB_MASTER_DATA, FIGURES_DIR, ensure_dirs_exist
)

# Ensure output directories exist
ensure_dirs_exist()

# Output paths
OUTPUT_STATUS_MAP = FIGURES_DIR / 'esb_adoption_status_map.png'
OUTPUT_INTENSITY_MAP = FIGURES_DIR / 'esb_adoption_intensity_map.png'
OUTPUT_YEAR_MAP = FIGURES_DIR / 'esb_adoption_by_year_map.png'

ID_COL = 'NCES District ID'

# --- 1. DATA LOADING AND CLEANING ---

def load_rebate_data(file_path):
    print(f"\nLoading Rebate data from {file_path}...")
    df = pd.read_excel(str(file_path))
    df['Project Status'] = df['Project Status'].astype(str).str.strip().str.upper()
    excluded_statuses = ['WITHDRAWN', 'CANCELLED', 'NOT SELECTED', 'INELIGIBLE', 'DENIED', 'nan']
    df = df[~df['Project Status'].isin(excluded_statuses)]
    
    df['IS_ADOPTER'] = 1
    df['NUM_ADOPTED'] = df['Number of Electric Buses']
    
    def get_mech(year):
        if str(year) == '2022': return 'LOTTERY_R1'
        if str(year) == '2023': return 'LOTTERY_R3'
        return 'LOTTERY_OTHER' 
    
    df['Funding Mechanism'] = df['Funding Year'].apply(get_mech)
    return df[[ID_COL, 'IS_ADOPTER', 'NUM_ADOPTED', 'Funding Mechanism']].drop_duplicates(subset=[ID_COL])

def load_grant_data(file_path):
    print(f"\nLoading Grant data from {file_path}...")
    df = pd.read_excel(str(file_path))
    if 'Electric Buses' in df.columns:
        df['NUM_ADOPTED'] = df['Electric Buses']
    else:
        df['NUM_ADOPTED'] = df['Project: Electric Buses']

    df['IS_ADOPTER'] = 1
    df['Funding Mechanism'] = 'COMPETITIVE_R2'
    
    if ID_COL not in df.columns:
        possible = [c for c in df.columns if 'NCES' in c and 'ID' in c]
        if possible:
            df.rename(columns={possible[0]: ID_COL}, inplace=True)
    
    return df[[ID_COL, 'IS_ADOPTER', 'NUM_ADOPTED', 'Funding Mechanism']].drop_duplicates(subset=[ID_COL])

# Load & Merge Data
df_rebates = load_rebate_data(ADOPTION_REBATES_FILE)
df_grants = load_grant_data(ADOPTION_GRANTS_FILE)
df_combined_winners = pd.concat([df_rebates, df_grants])

def aggregate_district(x):
    mechanisms = list(x['Funding Mechanism'].unique())
    if 'LOTTERY_R1' in mechanisms or 'LOTTERY_R3' in mechanisms:
        mech = 'LOTTERY_WINNER'
    elif 'COMPETITIVE_R2' in mechanisms:
        mech = 'COMPETITIVE_R2'
    else:
        mech = mechanisms[0]
    return pd.Series({
        'IS_ADOPTER': 1,
        'NUM_ADOPTED': x['NUM_ADOPTED'].sum(),
        'Funding Mechanism': mech
    })

df_winners = df_combined_winners.groupby(ID_COL).apply(aggregate_district).reset_index()

print("\nLoading Applicant Data...")
df_applicants = pd.read_excel(str(APPLICANT_FILE))
df_applicants['IS_APPLICANT'] = 1
df_applicants['Round'] = df_applicants['Round'].astype(str)

print("Merging Datasets...")
df_master = pd.merge(
    df_applicants[[ID_COL, 'IS_APPLICANT', 'Round']].drop_duplicates(subset=ID_COL),
    df_winners,
    on=ID_COL,
    how='outer'
)

df_master['IS_ADOPTER'] = df_master['IS_ADOPTER'].fillna(0).astype(int)
df_master['IS_APPLICANT'] = df_master['IS_APPLICANT'].fillna(0).astype(int)
df_master['NUM_ADOPTED'] = df_master['NUM_ADOPTED'].fillna(0)
df_master['Funding Mechanism'] = df_master['Funding Mechanism'].fillna('None')
df_master['IV_Z'] = df_master['Funding Mechanism'].isin(['LOTTERY_WINNER', 'LOTTERY_R1', 'LOTTERY_R3']).astype(int)
df_master[ID_COL] = df_master[ID_COL].astype(str).str.split('.').str[0].str.zfill(7)

# --- 2B. LOAD BUS-LEVEL DATA FOR YEAR-BASED ADOPTION MAP ---
print("\nLoading Bus-Level Data for Year-based Map...")
df_bus = pd.read_excel(str(WRI_EXCEL_FILE), sheet_name='2. Bus-level data')

# Extract Year from Quarter Awarded
df_bus['award_year'] = df_bus['3p. Quarter awarded'].str.extract(r'(\d{4})').astype(float)
df_bus['LEA_ID_clean'] = df_bus['1c. LEA ID'].astype(str).str.split('.').str[0].str.zfill(7)

# Get first adoption year per district and count buses
df_first_adoption = df_bus.groupby('LEA_ID_clean').agg({
    'award_year': 'min',  # First year adopted
    '3a. Number of ESBs committed ': 'sum'  # Total buses
}).reset_index()

df_first_adoption.rename(columns={
    'LEA_ID_clean': 'nces_id',
    'award_year': 'first_adoption_year',
    '3a. Number of ESBs committed ': 'total_esbs_from_bus_data'
}, inplace=True)

print(f"Districts with bus-level adoption data: {len(df_first_adoption)}")

# --- 3. MAPPING ---

try:
    print(f"Loading Shapefile...")
    districts_gdf = gpd.read_file(str(SCHOOL_DISTRICTS_SHP))
    shp_id_col = 'GEOID' if 'GEOID' in districts_gdf.columns else [c for c in districts_gdf.columns if 'ID' in c][0]
    districts_gdf[shp_id_col] = districts_gdf[shp_id_col].astype(str).str.zfill(7)

    print("Performing Spatial Join...")
    merged_gdf = districts_gdf.merge(df_master, left_on=shp_id_col, right_on=ID_COL, how='left')
    
    print("Filtering for Continental US (CONUS)...")
    merged_gdf['STATEFP'] = merged_gdf[shp_id_col].str[:2]
    conus_gdf = merged_gdf[~merged_gdf['STATEFP'].isin(['02', '15', '72', '78', '66', '60', '69'])].copy()
    conus_gdf = conus_gdf.to_crs(epsg=5070)
    conus_gdf['centroid'] = conus_gdf.geometry.centroid

    def get_status(row):
        if row['IS_ADOPTER'] == 1:
            if row['Funding Mechanism'] == 'LOTTERY_WINNER': return 'Lottery Winner (R1/R3)'
            if row['Funding Mechanism'] == 'COMPETITIVE_R2': return 'Grant Winner (R2)'
            return 'Other Adopter'
        elif row['IS_APPLICANT'] == 1: return 'Applicant (Waitlisted)'
        return 'Non-Applicant'

    conus_gdf['Map_Status'] = conus_gdf.apply(get_status, axis=1)

    colors = {
        'Non-Applicant': '#e0e0e0',
        'Applicant (Waitlisted)': '#ffcc00',
        'Grant Winner (R2)': '#56b4e9',
        'Lottery Winner (R1/R3)': '#005088'
    }

    # --- MAP 1: STATUS (POLYGONS) ---
    print("Generating Status Map (Polygons)...")
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    
    # 1. Plot Base
    base = conus_gdf[conus_gdf['Map_Status'] == 'Non-Applicant']
    if not base.empty:
        base.plot(ax=ax, color=colors['Non-Applicant'], linewidth=0.05, edgecolor='white')

    # 2. Plot Categories & Build Custom Legend
    plot_order = ['Applicant (Waitlisted)', 'Grant Winner (R2)', 'Lottery Winner (R1/R3)']
    legend_handles = []

    for cat in plot_order:
        subset = conus_gdf[conus_gdf['Map_Status'] == cat]
        if not subset.empty:
            color = colors.get(cat, 'grey')
            # Plot Polygons
            subset.plot(
                ax=ax,
                color=color,
                edgecolor='white',
                linewidth=0.1
            )
            # Create Custom Handle for Legend
            legend_handles.append(Patch(facecolor=color, edgecolor='white', label=cat))

    # 3. Add Manual Legend
    ax.legend(handles=legend_handles, loc='lower right', title="Adoption Status", fontsize=12)
    
    ax.set_title('ESB Adoption: Lottery vs. Competitive (CONUS)', fontsize=20)
    ax.set_axis_off()
    plt.savefig(str(OUTPUT_STATUS_MAP), dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_STATUS_MAP}")

    # --- MAP 2: INTENSITY (CENTROIDS/BUBBLES) ---
    print("Generating Intensity Map (Centroids)...")
    fig2, ax2 = plt.subplots(1, 1, figsize=(20, 12))
    
    # Base
    conus_gdf.plot(ax=ax2, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.05)
    
    # Adopters (Bubbles)
    adopters = conus_gdf[conus_gdf['IS_ADOPTER'] == 1].copy()
    if not adopters.empty:
        adopters.set_geometry('centroid').plot(
            ax=ax2,
            column='NUM_ADOPTED',
            cmap='viridis_r',
            markersize=adopters['NUM_ADOPTED'] * 8, # Scale size
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5,
            legend=True,
            legend_kwds={'label': "Number of Buses", 'orientation': "horizontal", 'shrink': 0.5, 'pad': 0.05}
        )

    ax2.set_title('ESB Adoption Intensity (Number of Buses)', fontsize=20)
    ax2.set_axis_off()
    plt.savefig(str(OUTPUT_INTENSITY_MAP), dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_INTENSITY_MAP}")

    # --- MAP 3: ADOPTION BY YEAR (CONTINENTAL US ONLY) ---
    print("Generating Adoption by Year Map (CONUS)...")
    
    # Merge bus-level data with geographic data
    year_gdf = districts_gdf.merge(df_first_adoption, left_on=shp_id_col, right_on='nces_id', how='left')
    year_gdf['STATEFP'] = year_gdf[shp_id_col].str[:2]
    year_gdf_conus = year_gdf[~year_gdf['STATEFP'].isin(['02', '15', '72', '78', '66', '60', '69'])].copy()
    year_gdf_conus = year_gdf_conus.to_crs(epsg=5070)
    
    fig3, ax3 = plt.subplots(1, 1, figsize=(20, 12))
    
    # Plot non-adopters in light gray
    non_adopters = year_gdf_conus[year_gdf_conus['first_adoption_year'].isna()]
    non_adopters.plot(ax=ax3, color='#f0f0f0', edgecolor='#d0d0d0', linewidth=0.05)
    
    # Plot adopters colored by year
    adopters_year = year_gdf_conus[year_gdf_conus['first_adoption_year'].notna()].copy()
    
    if not adopters_year.empty:
        adopters_year.plot(
            ax=ax3,
            column='first_adoption_year',
            cmap='YlOrRd',
            edgecolor='white',
            linewidth=0.1,
            legend=True,
            legend_kwds={
                'label': "Year of First ESB Adoption",
                'orientation': "horizontal",
                'shrink': 0.6,
                'pad': 0.05
            }
        )
    
    ax3.set_title('Electric School Bus Adoption Timeline (Continental US)', fontsize=20)
    ax3.set_axis_off()
    plt.savefig(str(OUTPUT_YEAR_MAP), dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_YEAR_MAP}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Save master dataset
df_master.to_csv(str(ESB_MASTER_DATA), index=False)
print(f"\nSaved master dataset to: {ESB_MASTER_DATA}")