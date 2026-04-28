"""Build the hazard-panel dataset used by the main Stata specs."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPT_DIR))

from config import CLEANED_DIR  # noqa: E402


def main() -> None:
    raise NotImplementedError("Expand district-level data into the hazard panel.")


if __name__ == "__main__":
    main()
