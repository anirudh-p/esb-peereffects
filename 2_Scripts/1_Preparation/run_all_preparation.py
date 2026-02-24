"""
run_all_preparation.py
======================
Master script to execute the complete data preparation pipeline.

Executes preparation scripts in correct dependency order:
    01 → Export WRI Excel to CSV
    02 → Extract charter flags
    03 → Prepare master regression data and maps
    04 → Create final analysis dataset with covariates
    05 → Build climate-based neighbor networks

Usage:
    python run_all_preparation.py [--skip-01] [--skip-02] [--skip-03] [--skip-04] [--skip-05]

Options:
    --skip-01    Skip WRI export (if CSV already exists)
    --skip-02    Skip charter flag extraction
    --skip-03    Skip data prep and mapping
    --skip-04    Skip analysis dataset creation
    --skip-05    Skip climate network construction

Author: [Your Name]
Last Updated: November 30, 2025
"""

import subprocess
import sys
from pathlib import Path
import time

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent

# Define pipeline scripts in execution order
PIPELINE = [
    ("01_export_wri_to_csv.py", "Export WRI data to CSV"),
    ("02_extract_charter_flags.py", "Extract charter school flags"),
    ("03_data_prep_and_map.py", "Prepare master regression data and create maps"),
    ("04_esb_analysis_prep.py", "Create final analysis dataset with covariates"),
    ("05_build_climate_networks.py", "Build climate-based neighbor networks"),
]

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    script_path = SCRIPT_DIR / script_name
    
    print("\n" + "="*80)
    print(f"RUNNING: {script_name}")
    print(f"Description: {description}")
    print("="*80)
    print(f"Script path: {script_path}\n")
    
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    try:
        start_time = time.time()
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SCRIPT_DIR.parent.parent),  # Run from project root
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ SUCCESS: {script_name} completed in {elapsed:.1f} seconds")
            return True
        else:
            print(f"\n✗ ERROR: {script_name} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n✗ EXCEPTION: {script_name} raised an error: {e}")
        return False

def main():
    """Run the complete data preparation pipeline."""
    print("\n" + "="*80)
    print("DATA PREPARATION PIPELINE")
    print("="*80)
    print(f"Total scripts: {len(PIPELINE)}")
    print(f"Working directory: {SCRIPT_DIR.parent.parent}")
    
    # Parse command line arguments for --skip flags
    skip_flags = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--skip-"):
            skip_num = arg.replace("--skip-", "")
            skip_flags.add(skip_num)
    
    if skip_flags:
        print(f"Skipping scripts: {', '.join(sorted(skip_flags))}")
    
    print("\nPipeline order:")
    for i, (script, desc) in enumerate(PIPELINE, 1):
        skip_marker = " [SKIP]" if f"{i:02d}" in skip_flags else ""
        print(f"  {i}. {script:<35} - {desc}{skip_marker}")
    
    input("\nPress Enter to start pipeline (or Ctrl+C to cancel)...")
    
    # Track results
    results = []
    start_time = time.time()
    
    # Execute each script
    for i, (script, description) in enumerate(PIPELINE, 1):
        script_num = f"{i:02d}"
        
        # Check if this script should be skipped
        if script_num in skip_flags:
            print(f"\n⊘ SKIPPING: {script} (--skip-{script_num} specified)")
            results.append((script, "SKIPPED"))
            continue
        
        success = run_script(script, description)
        results.append((script, "SUCCESS" if success else "FAILED"))
        
        if not success:
            print("\n" + "="*80)
            print("PIPELINE STOPPED DUE TO ERROR")
            print("="*80)
            print(f"Failed at: {script}")
            print("\nTo resume from a specific point, use --skip flags:")
            for j in range(1, i):
                print(f"  --skip-{j:02d}  (skip {PIPELINE[j-1][0]})")
            break
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"Total time: {total_time/60:.1f} minutes\n")
    
    for script, status in results:
        symbol = "✓" if status == "SUCCESS" else ("⊘" if status == "SKIPPED" else "✗")
        print(f"  {symbol} {script:<35} {status}")
    
    # Final status
    success_count = sum(1 for _, status in results if status == "SUCCESS")
    failed_count = sum(1 for _, status in results if status == "FAILED")
    skipped_count = sum(1 for _, status in results if status == "SKIPPED")
    
    print(f"\nResults: {success_count} succeeded, {failed_count} failed, {skipped_count} skipped")
    
    if failed_count == 0:
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETE!")
        print("="*80)
        print("\nAll cleaned data and weight matrices have been regenerated.")
        print("You can now run analysis scripts in 2_Analysis/")
        return 0
    else:
        print("\n" + "="*80)
        print("✗ PIPELINE INCOMPLETE")
        print("="*80)
        print(f"\n{failed_count} script(s) failed. Please fix errors and re-run.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user.")
        sys.exit(1)
