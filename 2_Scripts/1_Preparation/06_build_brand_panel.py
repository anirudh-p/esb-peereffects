import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import sys

print("Loading wide spatial panel and base dataset...")
df = pd.read_stata('1_Data/Cleaned/analysis_panel_dataset_spatial.dta')
base_df = pd.read_csv('1_Data/Cleaned/analysis_dataset.csv', dtype={'nces_id': str})

print("Loading K=6 spatial weights matrix to compute rival brand shocks...")
try:
    W6 = load_npz('1_Data/Cleaned/knn_weights_k6.npz')
except Exception as e:
    print(f"Could not load W6: {e}")
    sys.exit(1)

# Ensure order matches W6
order = pd.read_csv('1_Data/Cleaned/spatial_nces_order.csv', dtype={'nces_id': str})
order['w_index'] = np.arange(len(order))
base_ordered = order.merge(base_df, on='nces_id', how='left')

brands = ['Blue Bird', 'Lion Electric', 'Thomas Built Buses', 'IC Bus']
base_ordered['r1_brand'] = base_ordered['wri_first_oem'].apply(lambda x: x if x in brands else 'Other')

# Compute the static cross-sectional lag of R1 winners for each brand
brand_lags = {}
for b in brands + ['Other']:
    b_winners = ((base_ordered['IV_Z_R1'] == 1) & (base_ordered['r1_brand'] == b)).astype(float).values
    brand_lags[b] = W6.dot(b_winners)

# Join the cross-sectional static values into the panel
df['w_index'] = df['nces_id'].map(dict(zip(order['nces_id'], order['w_index'])))

for b in brands + ['Other']:
    # Use w_index to map the vector fast
    df[f'stat_Z_{b}'] = df['w_index'].map(lambda idx: brand_lags[b][int(idx)] if pd.notnull(idx) else 0)

# Merge focal district's eventual adopted brand
base_df['focal_brand'] = base_df['wri_first_oem'].apply(lambda x: x if x in brands else 'Other')
df = df.merge(base_df[['nces_id', 'focal_brand']], on='nces_id', how='left')

# Drop missing spatial obs to be safe
df = df.dropna(subset=['w_index'])

# Stack dataset
print("Stacking...")
stacked = []
for b in brands + ['Other']:
    temp = df.copy()
    temp['brand'] = b
    
    # Y is 1 if they adopted THIS brand in this year
    temp['Y_adopted_brand'] = ((temp['Y_first_adopted'] == 1) & (temp['focal_brand'] == b)).astype(int)
    
    # Z_peer_ib: Exposure to this exact brand (time-varying: only turns on post 2022)
    temp['Z_peer_brand_t_minus_1'] = np.where(temp['year'] > 2022, temp[f'stat_Z_{b}'], 0.0)
    
    # Z_peer_rival: Exposure to all other brands combined
    # Total Z minus the brand Z
    temp['Z_peer_rival_t_minus_1'] = np.where(temp['year'] > 2022, temp['w6_Z_t_minus_1'] - temp[f'stat_Z_{b}'], 0.0)
    
    stacked.append(temp)

df_stacked = pd.concat(stacked, ignore_index=True)

print(f"Stacked dataset shape: {df_stacked.shape}")
df_stacked.to_stata('1_Data/Cleaned/analysis_brand_panel_spatial.dta', write_index=False, version=118)
print("Finished stacking and saved to '1_Data/Cleaned/analysis_brand_panel_spatial.dta'!")
