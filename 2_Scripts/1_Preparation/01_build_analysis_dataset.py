"""Build the district-level base analysis dataset.

Start here. This script should harmonize identifiers, merge raw sources, and write
compact diagnostics before any hazard-panel expansion.
"""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPT_DIR))

from config import CLEANED_DIR  # noqa: E402


def main() -> None:
    raise NotImplementedError("Build district-level base dataset from raw sources.")


if __name__ == "__main__":
    main()
