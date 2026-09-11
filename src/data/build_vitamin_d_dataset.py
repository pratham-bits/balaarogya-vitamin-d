from pathlib import Path

import pandas as pd

from nhanes_loader import load_xpt


# ============================================================
# BalAarogya - Vitamin-D Module
# NHANES 2017-2018
# Step 2.5.4 - Build Analytical Dataset
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "vitamin_d_nhanes_2017_2018.csv"
)


# ------------------------------------------------------------
# Selected variables
# ------------------------------------------------------------

DEMO_COLUMNS = [
    "SEQN",
    "RIDAGEYR",
    "RIAGENDR",
]

BMX_COLUMNS = [
    "SEQN",
    "BMXWT",
]

DBQ_COLUMNS = [
    "SEQN",
    "DBQ197",
]

DR1TOT_COLUMNS = [
    "SEQN",
    "DR1TVD",
]

VID_COLUMNS = [
    "SEQN",
    "LBXVIDMS",
]


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

MIN_AGE = 1
MAX_AGE = 6


def load_selected_data() -> dict[str, pd.DataFrame]:
    """
    Load only the variables required for the first
    analytical dataset.
    """

    print("\n--- Loading selected NHANES variables ---")

    datasets = {
        "DEMO": load_xpt(
            "DEMO_J.XPT",
            DEMO_COLUMNS,
        ),
        "BMX": load_xpt(
            "BMX_J.XPT",
            BMX_COLUMNS,
        ),
        "DBQ": load_xpt(
            "DBQ_J.XPT",
            DBQ_COLUMNS,
        ),
        "DR1TOT": load_xpt(
            "DR1TOT_J.XPT",
            DR1TOT_COLUMNS,
        ),
        "VID": load_xpt(
            "VID_J.XPT",
            VID_COLUMNS,
        ),
    }

    for name, df in datasets.items():

        print(
            f"{name:8s} | "
            f"Rows: {len(df):,} | "
            f"Columns: {len(df.columns)}"
        )

    return datasets


def create_modeling_cohort(
    demo: pd.DataFrame,
    vid: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the supervised modeling cohort:

    Age 1-6
    +
    Valid laboratory total 25(OH)D measurement
    """

    print("\n--- Creating modeling cohort ---")

    # --------------------------------------------------------
    # Age restriction
    # --------------------------------------------------------

    child_demo = demo[
        demo["RIDAGEYR"].between(
            MIN_AGE,
            MAX_AGE,
            inclusive="both",
        )
    ].copy()

    print(
        f"Children aged {MIN_AGE}-{MAX_AGE}: "
        f"{len(child_demo):,}"
    )

    # --------------------------------------------------------
    # Valid laboratory target
    # --------------------------------------------------------

    valid_vid = vid[
        vid["LBXVIDMS"].notna()
    ].copy()

    print(
        "Participants with valid 25(OH)D: "
        f"{len(valid_vid):,}"
    )

    # --------------------------------------------------------
    # Intersection
    # --------------------------------------------------------

    child_seqn = set(
        child_demo["SEQN"]
    )

    valid_vid_seqn = set(
        valid_vid["SEQN"]
    )

    modeling_seqn = (
        child_seqn
        & valid_vid_seqn
    )

    print(
        "Final modeling cohort: "
        f"{len(modeling_seqn):,}"
    )

    # --------------------------------------------------------
    # Keep demographic records belonging to cohort
    # --------------------------------------------------------

    cohort = child_demo[
        child_demo["SEQN"].isin(
            modeling_seqn
        )
    ].copy()

    # --------------------------------------------------------
    # Add target
    # --------------------------------------------------------

    cohort = cohort.merge(
        valid_vid,
        on="SEQN",
        how="inner",
        validate="one_to_one",
    )

    return cohort


def merge_predictor_datasets(
    cohort: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge selected predictor datasets into the cohort.
    """

    print("\n--- Merging predictor datasets ---")

    merged = cohort.copy()

    merge_order = [
        "BMX",
        "DBQ",
        "DR1TOT",
    ]

    for name in merge_order:

        before_rows = len(merged)

        merged = merged.merge(
            datasets[name],
            on="SEQN",
            how="left",
            validate="one_to_one",
        )

        after_rows = len(merged)

        print(
            f"{name:8s} | "
            f"Rows before: {before_rows:,} | "
            f"Rows after: {after_rows:,}"
        )

        if after_rows != before_rows:

            raise RuntimeError(
                f"Unexpected row-count change after "
                f"merging {name}."
            )

    return merged


def clean_categorical_codes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply only clearly documented missing-code handling.

    No imputation is performed here.
    """

    print("\n--- Applying categorical missing-code rules ---")

    df = df.copy()

    # --------------------------------------------------------
    # DBQ197
    # --------------------------------------------------------
    # 7 = Refused
    # 9 = Don't know
    # Convert these to missing.
    #
    # Valid values 0-4 are retained.
    # --------------------------------------------------------

    if "DBQ197" in df.columns:

        df["DBQ197"] = df["DBQ197"].replace(
            {
                7.0: pd.NA,
                9.0: pd.NA,
            }
        )

    return df


def validate_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Perform structural validation on the final dataset.
    """

    print("\n" + "=" * 80)
    print("FINAL DATASET VALIDATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    print(
        f"\nRows: {len(df):,}"
    )

    if len(df) != 702:

        raise RuntimeError(
            "Expected exactly 702 rows, "
            f"but found {len(df)}."
        )

    # --------------------------------------------------------
    # SEQN uniqueness
    # --------------------------------------------------------

    duplicate_seqn = df[
        "SEQN"
    ].duplicated().sum()

    print(
        f"Duplicate SEQN: "
        f"{duplicate_seqn}"
    )

    if duplicate_seqn != 0:

        raise RuntimeError(
            "Duplicate SEQN detected."
        )

    # --------------------------------------------------------
    # Target validity
    # --------------------------------------------------------

    target_missing = df[
        "LBXVIDMS"
    ].isna().sum()

    print(
        f"Missing LBXVIDMS: "
        f"{target_missing}"
    )

    if target_missing != 0:

        raise RuntimeError(
            "Missing laboratory Vitamin-D target detected."
        )

    # --------------------------------------------------------
    # Age validation
    # --------------------------------------------------------

    invalid_age = (
        ~df["RIDAGEYR"].between(
            MIN_AGE,
            MAX_AGE,
            inclusive="both",
        )
    ).sum()

    print(
        f"Invalid age records: "
        f"{invalid_age}"
    )

    if invalid_age != 0:

        raise RuntimeError(
            "Age restriction validation failed."
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "BMXWT",
        "DBQ197",
        "DR1TVD",
        "LBXVIDMS",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Required columns missing: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Print missingness
    # --------------------------------------------------------

    print("\n--- Final missingness ---")

    for column in required_columns:

        missing = df[column].isna().sum()
        missing_pct = (
            df[column].isna().mean() * 100
        )

        print(
            f"{column:10s} | "
            f"Missing: {missing:3d} | "
            f"Missing %: {missing_pct:6.2f}%"
        )

    print("\nValidation PASSED.")


def save_dataset(
    df: pd.DataFrame,
) -> None:
    """Save analytical dataset to data/processed/."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nDataset saved to:\n"
        f"{OUTPUT_FILE}"
    )


def main() -> None:

    print("=" * 80)
    print("BalAarogya - Vitamin-D Module")
    print("NHANES 2017-2018")
    print("STEP 2.5.4 - BUILD ANALYTICAL DATASET")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load selected variables
    # --------------------------------------------------------

    datasets = load_selected_data()

    # --------------------------------------------------------
    # 2. Create 702-child cohort
    # --------------------------------------------------------

    cohort = create_modeling_cohort(
        datasets["DEMO"],
        datasets["VID"],
    )

    # --------------------------------------------------------
    # 3. Merge selected predictors
    # --------------------------------------------------------

    merged = merge_predictor_datasets(
        cohort,
        datasets,
    )

    # --------------------------------------------------------
    # 4. Apply minimal categorical cleaning
    # --------------------------------------------------------

    merged = clean_categorical_codes(
        merged
    )

    # --------------------------------------------------------
    # 5. Validate
    # --------------------------------------------------------

    validate_dataset(
        merged
    )

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    save_dataset(
        merged
    )

    print("\n" + "=" * 80)
    print("STEP 2.5.4 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()