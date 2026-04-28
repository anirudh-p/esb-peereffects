from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
STEPS = [
    SCRIPT_DIR / "1_Preparation" / "01_build_analysis_dataset.py",
    SCRIPT_DIR / "1_Preparation" / "02_build_panel_dataset.py",
    SCRIPT_DIR / "1_Preparation" / "03_build_spatial_design_exposure.py",
    SCRIPT_DIR / "1_Preparation" / "04_export_stata_inputs.py",
]

for step in STEPS:
    print(f"Running {step.relative_to(SCRIPT_DIR)}")
    subprocess.run([sys.executable, str(step)], check=True)
