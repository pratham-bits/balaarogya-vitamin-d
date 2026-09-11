from pathlib import Path

import pandas as pd


# ============================================================
# BalAarogya - Vitamin-D Module
# Step 2.5.5 - Analytical Dataset Schema Inspection
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vitamin_d_nhanes_2017_2018.csv"
)


EXPECTED_COLUMNS = [
    "SEQN",
    "RIDAGEYR",
    "RIAGENDR",
    "LBXVIDMS",
    "BMXWT",
    "DBQ197",
    "DR1TVD",
]


def main() -> None:

    print("=" * 80)
    print("BalAarogya - Vitamin-D Module")
    print("STEP 2.5.5 - ANALYTICAL DATASET SCHEMA INSPECTION")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("\n--- Loading analytical dataset ---")

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset path: {DATASET_PATH}")
    print(f"Shape: {df.shape}")

    # --------------------------------------------------------
    # 2. Column inspection
    # --------------------------------------------------------

    print("\n--- Columns ---")

    print(df.columns.tolist())

    # --------------------------------------------------------
    # 3. Verify expected columns
    # --------------------------------------------------------

    print("\n--- Expected column check ---")

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    unexpected_columns = [
        column
        for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    print(
        f"Missing expected columns: "
        f"{missing_columns}"
    )

    print(
        f"Unexpected columns: "
        f"{unexpected_columns}"
    )

    if missing_columns:
        raise RuntimeError(
            f"Required columns missing: {missing_columns}"
        )

    if unexpected_columns:
        raise RuntimeError(
            f"Unexpected columns found: {unexpected_columns}"
        )

    # --------------------------------------------------------
    # 4. Data types
    # --------------------------------------------------------

    print("\n--- Data types ---")

    print(df.dtypes)

    # --------------------------------------------------------
    # 5. First records
    # --------------------------------------------------------

    print("\n--- First 5 records ---")

    print(df.head().to_string(index=False))

    # --------------------------------------------------------
    # 6. Last records
    # --------------------------------------------------------

    print("\n--- Last 5 records ---")

    print(df.tail().to_string(index=False))

    # --------------------------------------------------------
    # 7. Duplicate SEQN
    # --------------------------------------------------------

    print("\n--- SEQN integrity ---")

    duplicate_seqn = df["SEQN"].duplicated().sum()

    print(
        f"Duplicate SEQN: {duplicate_seqn}"
    )

    if duplicate_seqn != 0:
        raise RuntimeError(
            "Duplicate SEQN detected."
        )

    print(
        f"Unique SEQN: "
        f"{df['SEQN'].nunique():,}"
    )

    # --------------------------------------------------------
    # 8. Age distribution
    # --------------------------------------------------------

    print("\n--- Age distribution ---")

    print(
        df["RIDAGEYR"]
        .value_counts()
        .sort_index()
    )

    print(
        f"Minimum age: "
        f"{df['RIDAGEYR'].min()}"
    )

    print(
        f"Maximum age: "
        f"{df['RIDAGEYR'].max()}"
    )

    # --------------------------------------------------------
    # 9. Sex distribution
    # --------------------------------------------------------

    print("\n--- Sex distribution ---")

    print(
        df["RIAGENDR"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # 10. DBQ197 distribution
    # --------------------------------------------------------

    print("\n--- DBQ197 distribution ---")

    print(
        df["DBQ197"]
        .value_counts(dropna=False)
        .sort_index()
    )

    # --------------------------------------------------------
    # 11. Missingness
    # --------------------------------------------------------

    print("\n--- Missingness ---")

    missingness = pd.DataFrame(
        {
            "Missing": df.isna().sum(),
            "Missing_%": (
                df.isna().mean() * 100
            ).round(2),
        }
    )

    print(missingness)

    # --------------------------------------------------------
    # 12. Target summary
    # --------------------------------------------------------

    print("\n--- Laboratory 25(OH)D target ---")

    print(
        df["LBXVIDMS"]
        .describe()
    )

    # --------------------------------------------------------
    # 13. Predictor summaries
    # --------------------------------------------------------

    print("\n--- Numeric predictor summaries ---")

    numeric_predictors = [
        "RIDAGEYR",
        "BMXWT",
        "DR1TVD",
    ]

    print(
        df[numeric_predictors]
        .describe()
    )

    # --------------------------------------------------------
    # 14. Basic range checks
    # --------------------------------------------------------

    print("\n--- Basic range checks ---")

    if not df["RIDAGEYR"].between(1, 6).all():
        raise RuntimeError(
            "Age range validation failed."
        )

    if not df["RIAGENDR"].isin([1, 2]).all():
        raise RuntimeError(
            "Unexpected sex code detected."
        )

    if (df["LBXVIDMS"] < 0).any():
        raise RuntimeError(
            "Negative 25(OH)D value detected."
        )

    if (df["BMXWT"] < 0).any():
        raise RuntimeError(
            "Negative weight detected."
        )

    if (df["DR1TVD"] < 0).any():
        raise RuntimeError(
            "Negative dietary Vitamin-D value detected."
        )

    print("Range checks PASSED.")

    # --------------------------------------------------------
    # 15. Final status
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("ANALYTICAL DATASET INSPECTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()