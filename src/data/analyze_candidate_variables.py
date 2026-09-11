from pathlib import Path

import pandas as pd


# ============================================================
# BalAarogya - Vitamin-D Module
# NHANES 2017-2018
# Step 2.5.1 - Candidate Variable Availability Analysis
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


DATASETS = {
    "VID": "VID_J.XPT",
    "DEMO": "DEMO_J.XPT",
    "BMX": "BMX_J.XPT",
    "DBQ": "DBQ_J.XPT",
    "DR1TOT": "DR1TOT_J.XPT",
    "DS1TOT": "DS1TOT_J.XPT",
}


# ------------------------------------------------------------
# Candidate variables
# ------------------------------------------------------------

CANDIDATE_VARIABLES = {
    "DEMO": [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "RIDRETH3",
    ],

    "BMX": [
        "BMXWT",
        "BMXHT",
        "BMXBMI",
    ],

    "DBQ": [
        "DBQ197",
        "DBQ223A",
        "DBQ223B",
        "DBQ223C",
        "DBQ223D",
        "DBQ223E",
        "DBQ223U",
    ],

    "DR1TOT": [
        "DR1TVD",
        "DR1TCALC",
        "DR1TPHOS",
    ],

    "DS1TOT": [
        "DS1DS",
        "DS1DSCNT",
        "DS1TVD",
        "DS1TCALC",
    ],

    "VID": [
        "LBXVIDMS",
    ],
}


def load_dataset(name: str) -> pd.DataFrame:
    """Load a complete NHANES XPT dataset."""
    
    file_path = RAW_DATA_DIR / DATASETS[name]

    return pd.read_sas(
        file_path,
        format="xport",
    )


def print_variable_analysis(
    dataset_name: str,
    df: pd.DataFrame,
    variables: list[str],
) -> None:
    """Print availability and basic statistics for selected variables."""

    print("\n" + "=" * 80)
    print(f"{dataset_name} DATASET")
    print("=" * 80)

    for variable in variables:

        if variable not in df.columns:
            print(f"\n{variable}")
            print("STATUS: VARIABLE NOT FOUND")
            continue

        series = df[variable]

        valid_count = series.notna().sum()
        missing_count = series.isna().sum()
        missing_pct = series.isna().mean() * 100

        print(f"\n--- {variable} ---")

        print(f"Data type       : {series.dtype}")
        print(f"Valid values    : {valid_count:,}")
        print(f"Missing values  : {missing_count:,}")
        print(f"Missing %       : {missing_pct:.2f}%")
        print(f"Unique values   : {series.nunique(dropna=True):,}")

        # ----------------------------------------------------
        # Numeric summary
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(series):

            print(f"Minimum         : {series.min()}")
            print(f"Maximum         : {series.max()}")
            print(f"Mean            : {series.mean()}")
            print(f"Median          : {series.median()}")

        # ----------------------------------------------------
        # Value counts
        # ----------------------------------------------------

        unique_values = series.dropna().value_counts().sort_index()

        if len(unique_values) <= 20:

            print("Value counts:")

            for value, count in unique_values.items():

                print(
                    f"    {value!r:<12} : {count:,}"
                )

        else:

            print(
                "Value counts    : "
                f"{len(unique_values):,} unique values "
                "(not displayed)"
            )


def main() -> None:

    print("=" * 80)
    print("BalAarogya - Vitamin-D Module")
    print("NHANES 2017-2018")
    print("STEP 2.5.1 - CANDIDATE VARIABLE AVAILABILITY ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load required datasets
    # --------------------------------------------------------

    print("\nLoading NHANES datasets...")

    datasets = {}

    for name in CANDIDATE_VARIABLES:

        print(f"Loading {name}...")

        datasets[name] = load_dataset(name)

    # --------------------------------------------------------
    # 2. Create the 702-child cohort
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CREATING 702-CHILD MODELING COHORT")
    print("=" * 80)

    demo = datasets["DEMO"]
    vid = datasets["VID"]

    # Children aged 1-6
    child_demo = demo[
        demo["RIDAGEYR"].between(
            1,
            6,
            inclusive="both",
        )
    ].copy()

    # Valid laboratory Vitamin-D measurement
    valid_vid = vid[
        vid["LBXVIDMS"].notna()
    ].copy()

    # Participant IDs
    child_seqn = set(child_demo["SEQN"])
    valid_vid_seqn = set(valid_vid["SEQN"])

    # Final cohort
    modeling_seqn = child_seqn & valid_vid_seqn

    print(
        f"\nChildren aged 1-6              : "
        f"{len(child_seqn):,}"
    )

    print(
        f"Children with valid 25(OH)D    : "
        f"{len(valid_vid_seqn):,}"
    )

    print(
        f"Final modeling cohort          : "
        f"{len(modeling_seqn):,}"
    )

    # --------------------------------------------------------
    # 3. Restrict every dataset to the modeling cohort
    # --------------------------------------------------------

    cohort_datasets = {}

    for name, df in datasets.items():

        cohort_df = df[
            df["SEQN"].isin(modeling_seqn)
        ].copy()

        cohort_datasets[name] = cohort_df

    # --------------------------------------------------------
    # 4. Analyze candidate variables
    # --------------------------------------------------------

    for name, variables in CANDIDATE_VARIABLES.items():

        print_variable_analysis(
            name,
            cohort_datasets[name],
            variables,
        )

    # --------------------------------------------------------
    # 5. Final summary table
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SUMMARY — VARIABLE AVAILABILITY IN 702-CHILD COHORT")
    print("=" * 80)

    summary_rows = []

    for dataset_name, variables in CANDIDATE_VARIABLES.items():

        df = cohort_datasets[dataset_name]

        for variable in variables:

            if variable not in df.columns:
                continue

            series = df[variable]

            valid_count = series.notna().sum()
            missing_count = series.isna().sum()

            summary_rows.append(
                {
                    "Dataset": dataset_name,
                    "Variable": variable,
                    "Valid": valid_count,
                    "Missing": missing_count,
                    "Missing_%": round(
                        series.isna().mean() * 100,
                        2,
                    ),
                    "Unique": series.nunique(
                        dropna=True
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 80)
    print("STEP 2.5.1 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()