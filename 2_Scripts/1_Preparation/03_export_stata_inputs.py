"""Export cleaned analysis files for Stata estimation."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(SCRIPT_DIR))

from config import CLEANED_DIR  # noqa: E402


def main() -> None:
    raise NotImplementedError("Export Stata-ready inputs from cleaned datasets.")


if __name__ == "__main__":
    main()
