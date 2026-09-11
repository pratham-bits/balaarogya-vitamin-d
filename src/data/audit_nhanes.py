from pathlib import Path

import pandas as pd


# ============================================================
# BalAarogya - Vitamin-D Module
# NHANES Dataset Audit
# ============================================================

# Project root:
# BalAarogya-VitaminD/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw NHANES data directory
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Interim output directory
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"

# NHANES files used for Vitamin-D module development
DATASETS = [
    "VID_J.XPT",
    "DEMO_J.XPT",
    "BMX_J.XPT",
    "DBQ_J.XPT",
    "DR1TOT_J.XPT",
    "DS1TOT_J.XPT",
]


def load_xpt(file_path: Path) -> pd.DataFrame:
    """Load an NHANES SAS Transport (.XPT) file."""
    return pd.read_sas(file_path, format="xport")


def audit_dataset(df: pd.DataFrame, file_name: str) -> None:
    """Print a concise audit summary for one dataset."""

    print("\n" + "=" * 80)
    print(f"DATASET: {file_name}")
    print("=" * 80)

    # Basic information
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]:,}")

    # --------------------------------------------------------
    # Column information
    # --------------------------------------------------------
    print("\n--- COLUMN INFORMATION ---")

    column_info = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing": df.isna().sum().values,
        "missing_%": (
            df.isna().mean().mul(100).round(2).values
        ),
        "unique": df.nunique(dropna=True).values,
    })

    print(column_info.to_string(index=False))

    # --------------------------------------------------------
    # Numeric summary
    # --------------------------------------------------------
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_columns) > 0:
        print("\n--- NUMERIC SUMMARY ---")

        numeric_summary = df[numeric_columns].describe().T

        print(
            numeric_summary[
                ["count", "mean", "std", "min", "max"]
            ].round(3).to_string()
        )


def main() -> None:
    print("=" * 80)
    print("BalAarogya - Vitamin-D Module")
    print("NHANES 2017-2018 DATASET AUDIT")
    print("=" * 80)

    # Check that the raw directory exists
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DATA_DIR}"
        )

    # Check that all expected datasets exist
    missing_files = [
        file_name
        for file_name in DATASETS
        if not (RAW_DATA_DIR / file_name).exists()
    ]

    if missing_files:
        print("\nERROR: The following files are missing:")
        for file_name in missing_files:
            print(f"  - {file_name}")

        raise FileNotFoundError(
            "Please place all required NHANES files in data/raw/."
        )

    # Audit every dataset
    for file_name in DATASETS:
        file_path = RAW_DATA_DIR / file_name

        print(f"\nLoading {file_name}...")

        df = load_xpt(file_path)

        audit_dataset(df, file_name)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()