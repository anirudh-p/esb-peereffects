"""
02b_build_radius_weights.py
Build radius-based matrices.
Outputs r{R}_{var}
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.sparse import save_npz, csr_matrix
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SHP_FILE, ANALYSIS_DATASET, CLEAN, LOGS_DIR, TABLES_DIR

def run():
    df = pd.read_csv(ANALYSIS_DATASET, dtype={'nces_id': str})
    shp = gpd.read_file(SHP_FILE)
    shp = shp.rename(columns={'GEOID': 'nces_id', 'LEAID': 'nces_id'})
    # Extract
    shp_c = shp.copy()
    shp_c['centroid'] = shp_c.geometry.centroid
    shp_c = shp_c.set_geometry('centroid')
    shp_c = shp_c.to_crs("EPSG:5070") # metres
    shp_c['cx'] = shp_c.geometry.x
    shp_c['cy'] = shp_c.geometry.y
    
    # Merge and subset
    df_r = df[df['nces_id'].isin(shp_c['nces_id'])].copy()
    shp_m = shp_c[shp_c['nces_id'].isin(df_r['nces_id'])].copy()
    
    shp_m = shp_m.drop_duplicates(subset=['nces_id'])
    df_r = shp_m[['nces_id']].merge(df_r, on='nces_id', how='left')
    
    coords = shp_m[['cx', 'cy']].values
    
    LAG_VARS = ["IV_Z_R1", "priority_r1", "IS_R1_LOSER"]
    
    base_file = CLEAN / "analysis_dataset_spatial.csv"
    if not base_file.exists():
        print("analysis_dataset_spatial.csv missing")
        return
    out_df = pd.read_csv(base_file, dtype={'nces_id': str})
    # ensure ordered properly
    df_r = df_r.merge(out_df[['nces_id', 'year_first_awarded', 'year_first_operating']], on='nces_id', how='left')
    
    RADIUS_MILES = [15, 30, 60]
    out_dict = {'nces_id': df_r['nces_id'].values}
    
    with open(LOGS_DIR / "radius_summary.txt", "w") as f:
        f.write("Radius Network Sizes\n=====================\n")
        
        for r_mi in RADIUS_MILES:
            r_m = r_mi * 1609.34
            nbrs = NearestNeighbors(radius=r_m, algorithm='ball_tree').fit(coords)
            adj_matrix = nbrs.radius_neighbors_graph(coords, mode='connectivity')
            
            # Remove self-loops
            adj_matrix.setdiag(0)
            adj_matrix.eliminate_zeros()
            
            # Save array
            save_npz(CLEAN / f"radius_weights_r{r_mi}.npz", adj_matrix)
            
            sums = adj_matrix.sum(axis=1).A1
            f.write(f"\nRadius {r_mi} miles:\n")
            f.write(f"  Mean neighbors: {sums.mean():.2f}\n")
            f.write(f"  Median neighbors: {np.median(sums):.2f}\n")
            f.write(f"  Min/Max: {sums.min():.0f} / {sums.max():.0f}\n")
            f.write(f"  Isolates (0 neighbors): {(sums == 0).sum()}\n")
            
            # Row standardize (just in case we need shares later)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row_sums = adj_matrix.sum(axis=1).A1
                row_sums[row_sums == 0] = 1.0
                W_rs = adj_matrix.multiply(1.0 / row_sums[:, None]).tocsr()
            
            for var in LAG_VARS:
                if var in df_r.columns:
                    x = df_r[var].fillna(0).values
                    # For radius, we just want counts typically
                    out_dict[f"r{r_mi}_{var}_count"] = adj_matrix.dot(x)
                    out_dict[f"r{r_mi}_{var}_share"] = W_rs.dot(x)
            
    radius_df = pd.DataFrame(out_dict)
    
    # Merge onto base out_df
    out_df = out_df.drop(columns=[c for c in radius_df.columns if c != 'nces_id' and c in out_df.columns])
    out_df = out_df.merge(radius_df, on='nces_id', how='left')
    out_df.to_csv(base_file, index=False)

    print("Radius computation done.")

if __name__ == "__main__":
    run()
