"""
Master script to generate all GIS map versions.
Runs all map generation scripts sequentially with progress tracking.
"""
import subprocess
import sys
import os
from pathlib import Path

HOME_PATH = r"C:\BC PhD\Research\Peer-Effects and Adoption"

# Map scripts to run
SCRIPTS = [
    ("Version 1: Adopters vs Non-Adopters", "map_v1_adopters.py"),
    ("Version 2: Number of ESBs", "map_v2_esb_count.py"),
    ("Version 3: District Size (Student Population)", "map_v3_district_size.py"),
    ("Version 4: Temporal Snapshots", "map_v4_temporal.py"),
]

def run_script(name, script_path):
    """Run a single map generation script."""
    print(f"\n{'='*80}")
    print(f"RUNNING: {name}")
    print(f"Script: {script_path}")
    print(f"{'='*80}\n")
    
    full_path = os.path.join(HOME_PATH, script_path)
    
    try:
        result = subprocess.run(
            [sys.executable, full_path],
            cwd=HOME_PATH,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per script
        )
        
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"\n❌ ERROR running {name}:")
            print(result.stderr)
            return False
        else:
            print(f"\n✓ SUCCESS: {name} completed")
            return True
            
    except subprocess.TimeoutExpired:
        print(f"\n❌ TIMEOUT: {name} took longer than 10 minutes")
        return False
    except Exception as e:
        print(f"\n❌ EXCEPTION running {name}: {e}")
        return False

def main():
    print("="*80)
    print("GIS MAP GENERATION - MASTER SCRIPT")
    print("="*80)
    print(f"Working directory: {HOME_PATH}")
    print(f"Total scripts to run: {len(SCRIPTS)}")
    print("="*80)
    
    results = {}
    
    for name, script in SCRIPTS:
        success = run_script(name, script)
        results[name] = success
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {name}")
    
    total = len(results)
    successes = sum(results.values())
    print(f"\nCompleted: {successes}/{total} scripts")
    
    if successes == total:
        print("\n🎉 All maps generated successfully!")
        print(f"\nOutput locations:")
        print(f"  - V1, V2, V3: {os.path.join(HOME_PATH, 'plots')}")
        print(f"  - V4 (Temporal): {os.path.join(HOME_PATH, 'plots', 'temporal_snapshots')}")
    else:
        print(f"\n⚠️  {total - successes} script(s) failed. Check output above for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
