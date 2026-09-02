"""
scripts/download_data.py
──────────────────────────────────────────────────────────────────────────────
Download PaySim dataset from Kaggle.

Prerequisites
─────────────
1. pip install kaggle
2. Place kaggle.json at ~/.kaggle/kaggle.json  (chmod 600 on Linux/Mac)
   OR set KAGGLE_USERNAME and KAGGLE_KEY environment variables.

Usage
─────
    python scripts/download_data.py
    python scripts/download_data.py --output-dir data/raw
"""
from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from loguru import logger


DATASET_SLUG = "ealaxi/paysim1"
OUTPUT_FILENAME = "PS_20174392719_1491204439457_log.csv"


def download(output_dir: Path) -> None:
    """Download and unzip the PaySim dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / OUTPUT_FILENAME

    if csv_path.exists():
        size_mb = csv_path.stat().st_size / 1e6
        logger.info(f"Dataset already exists at {csv_path} ({size_mb:.1f} MB). Skipping download.")
        return

    try:
        import kaggle  # noqa: F401 — validates credentials early
    except ImportError:
        raise ImportError("Run: pip install kaggle")
    except OSError as e:
        raise OSError(
            "Kaggle credentials not found. Place kaggle.json at ~/.kaggle/kaggle.json "
            "or set KAGGLE_USERNAME / KAGGLE_KEY env vars.\n"
            f"Original error: {e}"
        ) from e

    logger.info(f"Downloading {DATASET_SLUG} from Kaggle API (with automatic retries) …")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Download attempt {attempt}/{max_retries} …")
            api.dataset_download_files(DATASET_SLUG, path=str(output_dir), unzip=True)
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(3)

    if csv_path.exists():
        size_mb = csv_path.stat().st_size / 1e6
        logger.success(f"PaySim dataset ready at {csv_path} ({size_mb:.1f} MB)")
    else:
        # Check if CSV exists with slightly different filename
        csv_files = list(output_dir.glob("*.csv"))
        if csv_files:
            logger.success(f"Found CSV: {csv_files[0]}")
        else:
            raise FileNotFoundError(f"Download completed but no CSV found in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PaySim from Kaggle")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save the downloaded dataset",
    )
    args = parser.parse_args()
    download(args.output_dir)


if __name__ == "__main__":
    main()
