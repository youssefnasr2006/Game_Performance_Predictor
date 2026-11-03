"""Shared paths and data loading for the analysis scripts."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Datasets"
RAW_DATA = DATA_DIR / "raw_football_dataset.csv"
VERIFIED_DATA = DATA_DIR / "final_verified_dataset.csv"
FEATURED_DATA = DATA_DIR / "final_featured_dataset.csv"
CLEANED_DATA = DATA_DIR / "cleaned_raw_football_dataset.csv"


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA)


def load_verified() -> pd.DataFrame:
    df = pd.read_csv(VERIFIED_DATA)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    return df.sort_values(["Team", "date"]).reset_index(drop=True)
