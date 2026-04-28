"""Validate Stata-ready analysis inputs.

Earlier preparation scripts already write `.dta` mirrors for the district base,
hazard panel, and spatial/design panel. This step is intentionally small: it
checks that the expected Stata inputs exist and records their sizes.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPT_DIR))

from config import AUDIT_DIR, CLEANED_DIR  # noqa: E402

EXPECTED = [
    CLEANED_DIR / "analysis_district_base.dta",
    CLEANED_DIR / "analysis_hazard_panel.dta",
    CLEANED_DIR / "analysis_hazard_panel_spatial.dta",
]


def main() -> None:
    missing = [path for path in EXPECTED if not path.exists()]
    if missing:
        missing_list = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing expected Stata input(s):\n{missing_list}")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rows = ["file,size_bytes"]
    for path in EXPECTED:
        rows.append(f"{path.name},{path.stat().st_size}")
        print(f"Validated {path} ({path.stat().st_size:,} bytes)")
    (AUDIT_DIR / "stata_input_inventory.csv").write_text("\n".join(rows) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
