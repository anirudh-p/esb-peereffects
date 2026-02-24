# GIS Maps - Electric School Bus Adoption Visualization

This directory contains multiple Python scripts for visualizing electric school bus (ESB) adoption patterns across U.S. school districts.

## Overview

The scripts use:
- **Data source**: `merged.csv` - Bus-level information about ESB adoption in school districts
- **Shapefile**: EDGE_GEOCODE_PUBLICSCH_2223 - School district boundaries
- **Basemap**: CartoDB Positron (Google Maps style)
- **Libraries**: geopandas, matplotlib, contextily

## Map Versions

### Version 1: Adopters vs Non-Adopters
**File**: `map_v1_adopters.py`  
**Output**: `plots/map_v1_adopters_vs_non_adopters.png`

Shows binary adoption status:
- **Blue districts**: Have adopted electric school buses
- **Gray districts**: Have not adopted ESBs
- Includes count statistics and professional styling with basemap

**Usage**:
```powershell
python map_v1_adopters.py
```

### Version 2: Number of ESBs (Choropleth)
**File**: `map_v2_esb_count.py`  
**Output**: `plots/map_v2_esb_count.png`

Choropleth map showing fleet size:
- **Color gradient**: White to dark blue
- **Legend**: Shows number of electric buses per district
- Includes summary statistics (total ESBs, average per district, etc.)

**Usage**:
```powershell
python map_v2_esb_count.py
```

### Version 3: District Size (Student Population)
**File**: `map_v3_district_size.py`  
**Output**: `plots/map_v3_district_size.png`

Proportional symbol map:
- **Base layer**: All districts in light gray
- **Symbols**: Blue circles at district centroids
- **Circle size**: Proportional to student population (log scale)
- Shows which size districts are adopting ESBs

**Usage**:
```powershell
python map_v3_district_size.py
```

### Version 4: Temporal Snapshots
**File**: `map_v4_temporal.py`  
**Output**: `plots/temporal_snapshots/adoption_snapshot_YYYY.png`

Creates 5 maps showing cumulative adoption over time:
- **Years**: 2019, 2020, 2021, 2022, 2023
- Shows growth in adoption year-by-year
- Includes "new adopters this year" statistics

**Usage**:
```powershell
python map_v4_temporal.py
```

## Running All Maps

To generate all map versions at once:

```powershell
python generate_all_maps.py
```

This master script runs all four versions sequentially with progress tracking and error reporting.

## Output Directory Structure

```
plots/
├── map_v1_adopters_vs_non_adopters.png
├── map_v2_esb_count.png
├── map_v3_district_size.png
└── temporal_snapshots/
    ├── adoption_snapshot_2019.png
    ├── adoption_snapshot_2020.png
    ├── adoption_snapshot_2021.png
    ├── adoption_snapshot_2022.png
    └── adoption_snapshot_2023.png
```

## Data Columns Used

From `merged.csv`:
- `1c. LEA ID` - School district identifier (for joining with shapefile)
- `3b. Number of delivered or operating ESBs` - ESB count per district
- `4b. Number of students in district` - Student population
- `3p. Quarter awarded` - Award date (for temporal analysis)

## Dependencies

All required packages are listed in the environment. Key dependencies:
- `geopandas` - Geospatial data handling
- `matplotlib` - Plotting and visualization
- `contextily` - Basemap tiles
- `pandas` - Data manipulation

Install contextily if needed:
```powershell
python -m pip install contextily
```

## Customization

### Change basemap style
In any script, modify the `ctx.add_basemap()` call:
```python
# Options: CartoDB.Positron, CartoDB.Voyager, OpenStreetMap.Mapnik, etc.
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Voyager)
```

### Adjust map resolution
Change the `dpi` parameter in `plt.savefig()`:
```python
plt.savefig(OUTPUT_PATH, dpi=300, ...)  # Higher = better quality, larger file
```

### Zoom to specific region
Uncomment and adjust in any script:
```python
ax.set_xlim(-80, -72)  # Longitude bounds
ax.set_ylim(38, 43)    # Latitude bounds
```

## Performance Notes

- **Geometry simplification**: Applied (tolerance=1000m) for faster rendering
- **Column filtering**: Only essential columns loaded from shapefile
- **Typical runtime**: 1-3 minutes per map depending on basemap download
- **Output file size**: ~5-15 MB per PNG at 300 DPI

## Troubleshooting

**Basemap fails to load**:
- Check internet connection (basemaps are downloaded on-demand)
- Script will continue without basemap if it fails

**Memory issues**:
- Increase geometry simplification tolerance
- Reduce DPI for smaller file sizes

**Missing columns error**:
- Verify `merged.csv` contains required columns
- Auto-detection will attempt to find LEA ID column variants

## Original Script

The original `GIS_Maps.py` was the prototype. These new versions (v1-v4) are production-ready with:
- Better performance optimizations
- Professional styling
- Basemap integration
- Comprehensive error handling
- Multiple visualization types

## Author & Data

Maps created for BC PhD Research - Peer-Effects and Adoption study.
Data from EPA Clean School Bus Program (2017-2023).
