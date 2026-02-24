# ESB Adoption — Data Preparation and Peer Variable Construction

This document explains how the data was merged and summarizes the result, then documents how the LEA-level peer variables were created (k-nearest neighbors and within-radius metrics).

## Data sources and merge

- Workbook: `ESB_adoption_dataset_v9_update_june_2025.xlsx`
- Sheets used by default:
  - District-level: `1. District-level data`
  - Bus-level: `2. Bus-level data`
- Merge key: `1c. LEA ID` (present in both sheets).

We used the script `simple_merge.py` to perform a left merge of the bus-level sheet onto the district-level sheet on `1c. LEA ID`, and wrote `merged.csv`. Auto-detection confirmed that both sheets contained a column named `1c. LEA ID` and a full left-join match was achieved (i.e., all bus rows matched a district row).

### Merge summary (from a programmatic check)

- Bus-level rows: 12,549
- District-level rows: 19,517
- Merged rows (left-join with bus as left table): 12,549
- Unique LEA IDs in bus sheet: 1,559
- Unique LEA IDs in district sheet: 19,516
- Duplicate LEA IDs in district sheet: 0 (district LEA IDs are unique)
- Bus rows without a district match: 0

Note: District sheet provides geographic coordinates:
- Latitude: `1s. Latitude`
- Longitude: `1t. Longitude` (in some files appears with trailing whitespace; code trims column names)

These coordinates enable construction of inter-LEA distances for peer effects.

## LEA-level peer variables

Script: `Refinement.py`

Steps performed:
1. Load both sheets by name and trim column-name whitespace.
2. Build an LEA-level table by joining district rows with a bus-based adoption count:
   - `adoption_count`: number of bus rows linked to the LEA (from the bus sheet).
   - `adopted`: indicator = 1 if `adoption_count > 0`, else 0.
   - Rows missing coordinates are removed.
3. Compute pairwise proximity using a BallTree with the Haversine metric (coordinates in radians; Earth radius = 6371.0088 km).
4. Derive peer variables:
   - k-nearest neighbors (k=5 by default):
     - `knn5_peer_adopt_count`: number of adopted neighbors among the 5 nearest.
     - `knn5_peer_adopt_share`: unweighted share among the 5 nearest.
     - `knn5_peer_adopt_share_idw`: inverse-distance weighted share (with small epsilon to avoid division by zero).
   - Within-radius metrics (defaults: 50 km and 100 km):
     - `within_50km_peer_count`, `within_50km_peer_adopt_count`, `within_50km_peer_adopt_share`.
     - `within_100km_peer_count`, `within_100km_peer_adopt_count`, `within_100km_peer_adopt_share`.

### Output

- Output CSV: `lea_peer_vars.csv` (one row per LEA with coordinates, adoption flags, and peer variables).
- Example run summary:

```
rows: 19,516
unique_lea_ids: 19,516
missing_coords_rows: 0 (after dropping 1 row lacking coords)
columns_added:
  - knn5_peer_adopt_count
  - knn5_peer_adopt_share
  - knn5_peer_adopt_share_idw
  - within_50km_peer_count / adopt_count / adopt_share
  - within_100km_peer_count / adopt_count / adopt_share
```

## How to run

- Merge (optional; `merged.csv` already produced by `simple_merge.py`):

```powershell
python .\simple_merge.py
```

- Construct peer variables and save `lea_peer_vars.csv`:

```powershell
python .\Refinement.py --k 5 --radii 50 100
```

Arguments you can override:
- `--file`: alternate path to the Excel workbook.
- `--bus-sheet`, `--district-sheet`: alternate sheet names.
- `--lea-col`, `--lat-col`, `--lon-col`: alternate column names if your workbook differs.
- `--k`: number of nearest neighbors.
- `--radii`: one or more radii in km (space-separated) for within-radius metrics.

## Notes and caveats

- Adoption definition is “LEA has at least one row in the bus sheet,” which marks the presence of any electric school bus activity. You can refine this (e.g., delivered/operating only, year-specific) by filtering the bus sheet before aggregation.
- Coordinates are assumed to be district centroids in WGS84 (EPSG:4326). Haversine distances are used for great-circle distance in km.
- District entries labeled as non-LEA (e.g., private schools) appear in the district sheet; decide whether to include them depending on your research design.
- For dynamic peer effects, consider constructing time-lagged or event-study variants of the peer measures using bus timing fields (e.g., quarter delivered/operating).

## Stage timing imputation (interest, applied, awarded, operating)

Script: `Imputation.py`

Inputs:
- Workbook: `ESB_adoption_dataset_v9_update_june_2025.xlsx` (district + bus sheets)
- Calendar: `program_calendar.csv` (updated per program-year with application and award quarters)

What it does:
- Parses quarter strings like `YYYYQ#`.
- From bus-level data, computes earliest per-LEA quarters for `award`, `order`, `delivered`, `operating`.
- Infers application timing per LEA using waitlist cues (ARP/DERA), the program calendar, and, if needed, a coarse backcast from award quarter (−2Q).
- Sets an interest quarter using applied −1Q, else award −3Q when possible.
- Derives durations in quarters across stages, e.g., `award_to_order_q`, `order_to_delivered_q`, `delivered_to_operating_q`.

Output:
- `lea_stage_imputation.csv` with columns: `1c. LEA ID`, `interest_q`, `applied_q`, `awarded_q`, `order_q`, `delivered_q`, `operating_q`, provenance fields (`*_method`), and durations.

Run:

```powershell
python .\Imputation.py --file ESB_adoption_dataset_v9_update_june_2025.xlsx --calendar .\program_calendar.csv --out .\lea_stage_imputation.csv
```

Notes:
- Update `program_calendar.csv` as new funding rounds publish dates; rerun to refresh imputed fields.
- The provenance fields (`awarded_method`, `applied_method`, `interest_method`) indicate how each timestamp was set.

## LEA-by-quarter panel

Script: `BuildPanel.py`

What it does:
- Expands each LEA into the full quarter range observed in the imputed stages (global min to max quarter).
- For each LEA×quarter, constructs monotone stage flags: `interested`, `applied`, `awarded`, `ordered`, `delivered`, `operating`.
- Adds durations since stage (in quarters): `quarters_since_interest`, `quarters_since_applied`, `quarters_since_awarded`, `quarters_since_operating`.
- Attaches static attributes and peer variables from `lea_peer_vars.csv` (state, name, coordinates, peer counts/shares).

Output:
- `lea_quarter_panel.csv` with one row per LEA×quarter and the fields above, plus static attributes.

Run:

```powershell
python .\BuildPanel.py --stages .\lea_stage_imputation.csv --peer-vars .\lea_peer_vars.csv --out .\lea_quarter_panel.csv
```

Example build (latest run):
- Rows: ~1,014,884
- LEAs: 19,516
- Quarters: 52 (global min–max span in the data)

Next extensions:
- Compute dynamic peer adoption shares by quarter by building a neighbor map (from coordinates) and aggregating neighbors’ `awarded`/`operating` status each quarter.
- Add lagged peer measures (e.g., prior-quarter operating share) for regressions.
