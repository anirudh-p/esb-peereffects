"""
Generate comprehensive descriptive statistics for ESB adoption analysis.

Outputs:
    - 3_Output/Tables/Descriptive_Statistics/sample_summary.txt
    - 3_Output/Tables/Descriptive_Statistics/variable_statistics.txt
    - 3_Output/Tables/Descriptive_Statistics/adoption_by_category.txt
    - 3_Output/Tables/Descriptive_Statistics/correlation_matrix.txt

Author: Generated for ESB Analysis
Date: November 2025
"""

import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    ESB_FULL_ANALYSIS,
    CHARTER_FLAGS,
    TABLES_DIR,
    ensure_dirs_exist
)

import pandas as pd
import numpy as np

# ============================================================================
# SETUP
# ============================================================================

# Create output directory for descriptive statistics
DESCRIPTIVES_DIR = TABLES_DIR / "Descriptive_Statistics"
ensure_dirs_exist()
DESCRIPTIVES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DESCRIPTIVE STATISTICS GENERATION")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading data...")
df = pd.read_csv(str(ESB_FULL_ANALYSIS))
print(f"  Full sample: {len(df):,} districts")
print()

# ============================================================================
# SAMPLE SUMMARY
# ============================================================================

output_file = DESCRIPTIVES_DIR / "sample_summary.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SAMPLE SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    
    # Overall sample size
    f.write(f"Total Districts: {len(df):,}\n\n")
    
    # Charter vs traditional
    if 'is_charter' in df.columns:
        charter_counts = df['is_charter'].value_counts()
        n_traditional = charter_counts.get(0, 0)
        n_charter = charter_counts.get(1, 0)
        f.write(f"Traditional Districts: {n_traditional:,} ({n_traditional/len(df)*100:.1f}%)\n")
        f.write(f"Charter Schools: {n_charter:,} ({n_charter/len(df)*100:.1f}%)\n\n")
    
    # Adoption status
    if 'IS_ADOPTER' in df.columns:
        adoption_counts = df['IS_ADOPTER'].value_counts()
        n_adopted = adoption_counts.get(1, 0)
        n_not_adopted = adoption_counts.get(0, 0)
        f.write(f"Adopted ESB: {n_adopted:,} ({n_adopted/len(df)*100:.1f}%)\n")
        f.write(f"Did Not Adopt: {n_not_adopted:,} ({n_not_adopted/len(df)*100:.1f}%)\n\n")
        
        # Adoption by charter status
        if 'is_charter' in df.columns:
            f.write("\nAdoption by District Type:\n")
            f.write("-" * 40 + "\n")
            crosstab = pd.crosstab(df['is_charter'], df['IS_ADOPTER'], 
                                   normalize='index') * 100
            crosstab.index = ['Traditional', 'Charter']
            crosstab.columns = ['No Adoption', 'Adopted']
            f.write(crosstab.to_string(float_format=lambda x: f"{x:.1f}%"))
            f.write("\n\n")
    
    # Application status
    if 'IS_APPLICANT' in df.columns:
        app_counts = df['IS_APPLICANT'].value_counts()
        n_applicants = app_counts.get(1, 0)
        f.write(f"Applied for Funding: {n_applicants:,} ({n_applicants/len(df)*100:.1f}%)\n")
        
        if 'IS_LOTTERY_WINNER' in df.columns:
            winners = df[df['IS_APPLICANT'] == 1]['IS_LOTTERY_WINNER'].sum()
            f.write(f"  Winners: {winners:,} ({winners/n_applicants*100:.1f}% of applicants)\n")
            losers = n_applicants - winners
            f.write(f"  Losers: {losers:,} ({losers/n_applicants*100:.1f}% of applicants)\n\n")
    
    # Priority status (among applicants)
    if 'is_priority' in df.columns and 'IS_APPLICANT' in df.columns:
        priority_data = df[df['IS_APPLICANT'] == 1]['is_priority']
        n_priority = priority_data.sum()
        n_non_priority = len(priority_data) - n_priority
        f.write(f"\nPriority Status (Among Applicants):\n")
        f.write(f"  Priority Districts: {n_priority:,} ({n_priority/len(priority_data)*100:.1f}%)\n")
        f.write(f"  Non-Priority: {n_non_priority:,} ({n_non_priority/len(priority_data)*100:.1f}%)\n\n")
    
    # Urbanicity distribution
    if 'urbanicity' in df.columns:
        f.write("\nUrbanicity Distribution:\n")
        f.write("-" * 40 + "\n")
        urban_counts = df['urbanicity'].value_counts().sort_index()
        for locale, count in urban_counts.items():
            f.write(f"  {locale}: {count:,} ({count/len(df)*100:.1f}%)\n")
        f.write("\n")
    
    # Geographic coverage
    if 'state' in df.columns:
        n_states = df['state'].nunique()
        f.write(f"States Represented: {n_states}\n\n")

print(f"✓ Sample summary saved to: {output_file}")
print()

# ============================================================================
# VARIABLE STATISTICS
# ============================================================================

# Define key variables for descriptive stats
key_vars = {
    'median_income': 'Median Household Income ($)',
    'poverty_rate': 'Poverty Rate',
    'enrollment': 'Total Enrollment',
    'log_enrollment': 'Log(Enrollment)',
    'pm25': 'PM2.5 (ug/m3)',
    'pct_white': 'Percent White',
    'pct_black': 'Percent Black',
    'pct_hispanic': 'Percent Hispanic',
    'fleet_size': 'Bus Fleet Size',
    'IS_ADOPTER': 'Adopted ESB (0/1)',
}

# Filter to variables that exist in the dataset
available_vars = {k: v for k, v in key_vars.items() if k in df.columns}

output_file = DESCRIPTIVES_DIR / "variable_statistics.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("VARIABLE STATISTICS (FULL SAMPLE)\n")
    f.write("=" * 80 + "\n\n")
    
    stats_df = pd.DataFrame()
    for var, label in available_vars.items():
        data = df[var].dropna()
        stats_df[var] = [
            label,
            len(data),
            data.mean(),
            data.std(),
            data.min(),
            data.quantile(0.25),
            data.median(),
            data.quantile(0.75),
            data.max()
        ]
    
    stats_df.index = ['Variable', 'N', 'Mean', 'Std Dev', 'Min', 
                       '25th Pct', 'Median', '75th Pct', 'Max']
    
    f.write(stats_df.T.to_string())
    f.write("\n\n")
    
    # Statistics by charter status
    if 'is_charter' in df.columns:
        f.write("=" * 80 + "\n")
        f.write("VARIABLE STATISTICS BY DISTRICT TYPE\n")
        f.write("=" * 80 + "\n\n")
        
        for district_type, district_label in [(0, 'TRADITIONAL DISTRICTS'), 
                                                (1, 'CHARTER SCHOOLS')]:
            subset = df[df['is_charter'] == district_type]
            f.write(f"\n{district_label} (N = {len(subset):,})\n")
            f.write("-" * 80 + "\n")
            
            stats_df_sub = pd.DataFrame()
            for var, label in available_vars.items():
                if var == 'IS_ADOPTER':  # Skip adoption for separate analysis
                    continue
                data = subset[var].dropna()
                if len(data) > 0:
                    stats_df_sub[var] = [
                        label,
                        len(data),
                        data.mean(),
                        data.std(),
                        data.median()
                    ]
            
            stats_df_sub.index = ['Variable', 'N', 'Mean', 'Std Dev', 'Median']
            f.write(stats_df_sub.T.to_string())
            f.write("\n")

print(f"✓ Variable statistics saved to: {output_file}")
print()

# ============================================================================
# ADOPTION BY CATEGORIES
# ============================================================================

output_file = DESCRIPTIVES_DIR / "adoption_by_category.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ESB ADOPTION BY CATEGORICAL VARIABLES\n")
    f.write("=" * 80 + "\n\n")
    
    # By urbanicity
    if 'urbanicity' in df.columns and 'IS_ADOPTER' in df.columns:
        f.write("Adoption Rate by Urbanicity:\n")
        f.write("-" * 40 + "\n")
        urban_adopt = df.groupby('urbanicity')['IS_ADOPTER'].agg(['sum', 'count', 'mean'])
        urban_adopt.columns = ['Adopted', 'Total', 'Rate']
        urban_adopt['Rate'] = urban_adopt['Rate'] * 100
        f.write(urban_adopt.to_string(float_format=lambda x: f"{x:.1f}" if x < 10 else f"{x:.0f}"))
        f.write("\n\n")
    
    # By application/winner status
    if all(col in df.columns for col in ['IS_APPLICANT', 'IS_ADOPTER']):
        f.write("Adoption Rate by Application Status:\n")
        f.write("-" * 40 + "\n")
        app_adopt = df.groupby('IS_APPLICANT')['IS_ADOPTER'].agg(['sum', 'count', 'mean'])
        app_adopt.index = ['Non-Applicant', 'Applicant']
        app_adopt.columns = ['Adopted', 'Total', 'Rate']
        app_adopt['Rate'] = app_adopt['Rate'] * 100
        f.write(app_adopt.to_string(float_format=lambda x: f"{x:.1f}" if x < 10 else f"{x:.0f}"))
        f.write("\n\n")
        
        # Among applicants: winners vs losers
        if 'IS_LOTTERY_WINNER' in df.columns:
            applicants = df[df['IS_APPLICANT'] == 1]
            f.write("Adoption Rate Among Applicants (Winners vs Losers):\n")
            f.write("-" * 40 + "\n")
            winner_adopt = applicants.groupby('IS_LOTTERY_WINNER')['IS_ADOPTER'].agg(['sum', 'count', 'mean'])
            winner_adopt.index = ['Loser', 'Winner']
            winner_adopt.columns = ['Adopted', 'Total', 'Rate']
            winner_adopt['Rate'] = winner_adopt['Rate'] * 100
            f.write(winner_adopt.to_string(float_format=lambda x: f"{x:.1f}" if x < 10 else f"{x:.0f}"))
            f.write("\n\n")
    
    # By priority status (among applicants)
    if all(col in df.columns for col in ['IS_APPLICANT', 'is_priority', 'IS_ADOPTER']):
        applicants = df[df['IS_APPLICANT'] == 1]
        f.write("Adoption Rate by Priority Status (Among Applicants):\n")
        f.write("-" * 40 + "\n")
        priority_adopt = applicants.groupby('is_priority')['IS_ADOPTER'].agg(['sum', 'count', 'mean'])
        priority_adopt.index = ['Non-Priority', 'Priority']
        priority_adopt.columns = ['Adopted', 'Total', 'Rate']
        priority_adopt['Rate'] = priority_adopt['Rate'] * 100
        f.write(priority_adopt.to_string(float_format=lambda x: f"{x:.1f}" if x < 10 else f"{x:.0f}"))
        f.write("\n\n")
    
    # By state (top 15)
    if 'state' in df.columns and 'IS_ADOPTER' in df.columns:
        f.write("Adoption Rate by State (Top 15 by Sample Size):\n")
        f.write("-" * 40 + "\n")
        state_adopt = df.groupby('state')['IS_ADOPTER'].agg(['sum', 'count', 'mean'])
        state_adopt.columns = ['Adopted', 'Total', 'Rate']
        state_adopt['Rate'] = state_adopt['Rate'] * 100
        state_adopt = state_adopt.sort_values('Total', ascending=False).head(15)
        f.write(state_adopt.to_string(float_format=lambda x: f"{x:.1f}" if x < 10 else f"{x:.0f}"))
        f.write("\n\n")

print(f"✓ Adoption by category saved to: {output_file}")
print()

# ============================================================================
# CORRELATION MATRIX
# ============================================================================

output_file = DESCRIPTIVES_DIR / "correlation_matrix.txt"

# Select numeric variables for correlation
corr_vars = ['IS_ADOPTER', 'median_income', 'poverty_rate', 'log_enrollment', 
             'pm25', 'pct_white', 'pct_black', 'pct_hispanic', 'fleet_size']
corr_vars = [v for v in corr_vars if v in df.columns]

if len(corr_vars) > 1:
    corr_matrix = df[corr_vars].corr()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CORRELATION MATRIX (KEY VARIABLES)\n")
        f.write("=" * 80 + "\n\n")
        f.write(corr_matrix.to_string(float_format=lambda x: f"{x:.3f}"))
        f.write("\n\n")
        
        # Highlight strong correlations with adoption
        if 'IS_ADOPTER' in corr_vars:
            f.write("Correlations with ESB Adoption:\n")
            f.write("-" * 40 + "\n")
            adopt_corr = corr_matrix['IS_ADOPTER'].drop('IS_ADOPTER').sort_values(ascending=False)
            for var, corr in adopt_corr.items():
                f.write(f"{var:20s}: {corr:6.3f}\n")
    
    print(f"✓ Correlation matrix saved to: {output_file}")
    print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("DESCRIPTIVE STATISTICS COMPLETE")
print("=" * 80)
print(f"\nAll tables saved to: {DESCRIPTIVES_DIR}")
print()
print("Generated files:")
print("  - sample_summary.txt")
print("  - variable_statistics.txt")
print("  - adoption_by_category.txt")
print("  - correlation_matrix.txt")
print()
