from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer


# =============================================================================
# BalAarogya - Vitamin-D Module
# PHASE 3.3 - MISSING DATA & PREPROCESSING
# SUB-PHASE 3.3.2 - IMPUTATION STRATEGY COMPARISON
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vitamin_d_nhanes_2017_2018.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eda"
    / "imputation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET = "LBXVIDMS"

NUMERIC_FEATURES = [
    "BMXWT",
    "DR1TVD",
]

FEATURES_WITHOUT_DIETARY_VITAMIN_D = [
    "RIDAGEYR",
    "RIAGENDR",
    "BMXWT",
    "DBQ197",
]

PRIMARY_FEATURES = [
    "RIDAGEYR",
    "RIAGENDR",
    "BMXWT",
    "DBQ197",
    "DR1TVD",
]


# =============================================================================
# LOAD DATA
# =============================================================================

df = pd.read_csv(DATA_PATH)


print("=" * 80)
print("BalAarogya - Vitamin-D Module")
print("PHASE 3.3 - MISSING DATA & PREPROCESSING")
print("SUB-PHASE 3.3.2 - IMPUTATION STRATEGY COMPARISON")
print("=" * 80)


print("\n" + "=" * 80)
print("DATASET")
print("=" * 80)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Target: {TARGET}")
print(f"Target missing: {df[TARGET].isna().sum()}")


# =============================================================================
# ORIGINAL MISSINGNESS
# =============================================================================

print("\n" + "=" * 80)
print("ORIGINAL MISSINGNESS")
print("=" * 80)

original_missing = pd.DataFrame(
    {
        "Feature": NUMERIC_FEATURES,
        "Missing_Count": [
            df[col].isna().sum()
            for col in NUMERIC_FEATURES
        ],
        "Missing_Percentage": [
            df[col].isna().mean() * 100
            for col in NUMERIC_FEATURES
        ],
    }
)

print(original_missing.to_string(index=False))

original_missing.to_csv(
    OUTPUT_DIR / "original_missingness.csv",
    index=False,
)


# =============================================================================
# STRATEGY A - COMPLETE CASE
# =============================================================================

print("\n" + "=" * 80)
print("STRATEGY A - COMPLETE CASE")
print("=" * 80)

complete_case_df = df.dropna(subset=PRIMARY_FEATURES).copy()

complete_case_n = len(complete_case_df)

print(f"Complete-case rows: {complete_case_n}")
print(f"Rows removed: {len(df) - complete_case_n}")
print(
    f"Percentage of original cohort retained: "
    f"{complete_case_n / len(df) * 100:.2f}%"
)

print("\nRemaining missing values:")

print(
    complete_case_df[PRIMARY_FEATURES]
    .isna()
    .sum()
    .to_string()
)


# =============================================================================
# STRATEGY B - MEDIAN IMPUTATION
# =============================================================================

print("\n" + "=" * 80)
print("STRATEGY B - MEDIAN IMPUTATION")
print("=" * 80)

median_imputed_df = df.copy()

median_values = {}

for feature in NUMERIC_FEATURES:
    median_value = median_imputed_df[feature].median()
    median_values[feature] = median_value

    median_imputed_df[feature] = (
        median_imputed_df[feature]
        .fillna(median_value)
    )

    print(
        f"{feature}: "
        f"median used = {median_value:.4f}"
    )


print("\nRemaining missing values:")

print(
    median_imputed_df[NUMERIC_FEATURES]
    .isna()
    .sum()
    .to_string()
)


median_summary = pd.DataFrame(
    {
        "Feature": list(median_values.keys()),
        "Median_Used": list(median_values.values()),
    }
)

median_summary.to_csv(
    OUTPUT_DIR / "median_imputation_values.csv",
    index=False,
)


# =============================================================================
# STRATEGY C - MEDIAN + MISSINGNESS INDICATOR
# =============================================================================

print("\n" + "=" * 80)
print("STRATEGY C - MEDIAN IMPUTATION + MISSINGNESS INDICATOR")
print("=" * 80)

indicator_df = df.copy()

for feature in NUMERIC_FEATURES:

    indicator_name = f"{feature}_missing"

    indicator_df[indicator_name] = (
        indicator_df[feature]
        .isna()
        .astype(int)
    )

    median_value = indicator_df[feature].median()

    indicator_df[feature] = (
        indicator_df[feature]
        .fillna(median_value)
    )

    print(
        f"{feature}: "
        f"median = {median_value:.4f}, "
        f"indicator = {indicator_name}"
    )


indicator_columns = [
    f"{feature}_missing"
    for feature in NUMERIC_FEATURES
]

print("\nMissingness indicator distributions:")

for column in indicator_columns:
    print(f"\n{column}")
    print(indicator_df[column].value_counts().sort_index().to_string())


print("\nRemaining missing values:")

print(
    indicator_df[NUMERIC_FEATURES]
    .isna()
    .sum()
    .to_string()
)


# =============================================================================
# STRATEGY D - EXCLUDE DR1TVD
# =============================================================================

print("\n" + "=" * 80)
print("STRATEGY D - EXCLUDE DR1TVD")
print("=" * 80)

no_dietary_df = df[FEATURES_WITHOUT_DIETARY_VITAMIN_D].copy()

print("DR1TVD is excluded completely.")

print(f"Rows available before BMXWT handling: {len(no_dietary_df)}")

bmxwt_median = no_dietary_df["BMXWT"].median()

no_dietary_df["BMXWT"] = (
    no_dietary_df["BMXWT"]
    .fillna(bmxwt_median)
)

print(f"BMXWT median used: {bmxwt_median:.4f}")

print("\nRemaining missing values:")

print(
    no_dietary_df
    .isna()
    .sum()
    .to_string()
)


# =============================================================================
# STRATEGY SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("STRATEGY SUMMARY")
print("=" * 80)

strategy_summary = pd.DataFrame(
    [
        {
            "Strategy": "A_Complete_Case",
            "DR1TVD_Used": True,
            "Imputation": "None",
            "Missingness_Indicator": False,
            "Rows_Available": len(complete_case_df),
            "Rows_Retained_Percent": (
                len(complete_case_df) / len(df) * 100
            ),
        },
        {
            "Strategy": "B_Median",
            "DR1TVD_Used": True,
            "Imputation": "Median",
            "Missingness_Indicator": False,
            "Rows_Available": len(median_imputed_df),
            "Rows_Retained_Percent": (
                len(median_imputed_df) / len(df) * 100
            ),
        },
        {
            "Strategy": "C_Median_Indicator",
            "DR1TVD_Used": True,
            "Imputation": "Median",
            "Missingness_Indicator": True,
            "Rows_Available": len(indicator_df),
            "Rows_Retained_Percent": (
                len(indicator_df) / len(df) * 100
            ),
        },
        {
            "Strategy": "D_Exclude_DR1TVD",
            "DR1TVD_Used": False,
            "Imputation": "Median for BMXWT",
            "Missingness_Indicator": False,
            "Rows_Available": len(no_dietary_df),
            "Rows_Retained_Percent": (
                len(no_dietary_df) / len(df) * 100
            ),
        },
    ]
)

print(strategy_summary.to_string(index=False))

strategy_summary.to_csv(
    OUTPUT_DIR / "imputation_strategy_summary.csv",
    index=False,
)


# =============================================================================
# DISTRIBUTION COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("DISTRIBUTION COMPARISON")
print("=" * 80)


distribution_rows = []

for feature in NUMERIC_FEATURES:

    original = df[feature].dropna()

    median_value = df[feature].median()

    imputed = df[feature].fillna(median_value)

    distribution_rows.append(
        {
            "Feature": feature,
            "Observed_N": original.count(),
            "Original_N": len(df),
            "Missing_N": df[feature].isna().sum(),
            "Original_Mean_Observed": original.mean(),
            "Original_Median_Observed": original.median(),
            "Original_Std_Observed": original.std(),
            "Imputed_Mean": imputed.mean(),
            "Imputed_Median": imputed.median(),
            "Imputed_Std": imputed.std(),
            "Median_Used": median_value,
        }
    )


distribution_summary = pd.DataFrame(distribution_rows)

print(
    distribution_summary.to_string(index=False)
)

distribution_summary.to_csv(
    OUTPUT_DIR / "distribution_before_after_imputation.csv",
    index=False,
)


# =============================================================================
# IMPUTED VALUE COUNTS
# =============================================================================

print("\n" + "=" * 80)
print("NUMBER OF VALUES REPLACED BY MEDIAN")
print("=" * 80)

for feature in NUMERIC_FEATURES:

    missing_count = df[feature].isna().sum()

    print(
        f"{feature}: "
        f"{missing_count} values replaced"
    )


# =============================================================================
# PLOT 1 - OBSERVED VS MEDIAN-IMPUTED DISTRIBUTION
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING IMPUTATION DIAGNOSTIC PLOTS")
print("=" * 80)


for feature in NUMERIC_FEATURES:

    observed = df[feature].dropna()

    median_value = df[feature].median()

    imputed = df[feature].fillna(median_value)

    plt.figure(figsize=(10, 6))

    plt.hist(
        observed,
        bins=20,
        alpha=0.7,
        label="Observed",
        edgecolor="black",
    )

    plt.hist(
        imputed,
        bins=20,
        alpha=0.35,
        label="After median imputation",
        edgecolor="black",
    )

    plt.axvline(
        median_value,
        linestyle="--",
        label=f"Median = {median_value:.2f}",
    )

    plt.xlabel(feature)
    plt.ylabel("Number of children")
    plt.title(
        f"{feature}: Observed vs Median-Imputed Distribution"
    )

    plt.legend()
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{feature.lower()}_observed_vs_imputed.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")


# =============================================================================
# PLOT 2 - MISSINGNESS INDICATOR COUNTS
# =============================================================================

plt.figure(figsize=(8, 6))

indicator_counts = [
    df[feature].isna().sum()
    for feature in NUMERIC_FEATURES
]

observed_counts = [
    df[feature].notna().sum()
    for feature in NUMERIC_FEATURES
]

x_positions = range(len(NUMERIC_FEATURES))

plt.bar(
    x_positions,
    observed_counts,
    label="Observed",
)

plt.bar(
    x_positions,
    indicator_counts,
    bottom=observed_counts,
    label="Missing",
)

plt.xticks(
    list(x_positions),
    NUMERIC_FEATURES,
)

plt.ylabel("Number of children")
plt.xlabel("Feature")
plt.title("Observed vs Missing Values")

plt.legend()
plt.tight_layout()

missingness_plot = (
    OUTPUT_DIR
    / "observed_vs_missing_counts.png"
)

plt.savefig(
    missingness_plot,
    dpi=150,
)

plt.close()

print(f"Saved: {missingness_plot}")


# =============================================================================
# PLOT 3 - STRATEGY SAMPLE RETENTION
# =============================================================================

plt.figure(figsize=(10, 6))

strategy_names = strategy_summary["Strategy"]
strategy_rows = strategy_summary["Rows_Available"]

plt.bar(
    range(len(strategy_names)),
    strategy_rows,
)

plt.xticks(
    range(len(strategy_names)),
    strategy_names,
    rotation=20,
    ha="right",
)

plt.ylabel("Rows available")
plt.xlabel("Imputation strategy")
plt.title("Sample Retention by Imputation Strategy")

plt.tight_layout()

retention_plot = (
    OUTPUT_DIR
    / "imputation_strategy_sample_retention.png"
)

plt.savefig(
    retention_plot,
    dpi=150,
)

plt.close()

print(f"Saved: {retention_plot}")


# =============================================================================
# FINAL VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

assert df[TARGET].notna().all(), (
    "Target contains missing values."
)

assert len(complete_case_df) < len(df), (
    "Complete-case strategy should remove some rows."
)

assert median_imputed_df[NUMERIC_FEATURES].notna().all().all(), (
    "Median-imputed dataset still contains missing numeric values."
)

assert indicator_df[NUMERIC_FEATURES].notna().all().all(), (
    "Indicator dataset still contains missing numeric values."
)

assert no_dietary_df["BMXWT"].notna().all(), (
    "BMXWT remains missing in DR1TVD-excluded strategy."
)

assert len(strategy_summary) == 4, (
    "Expected four imputation strategies."
)

print("All validation checks passed.")


# =============================================================================
# COMPLETION
# =============================================================================

print("\n" + "=" * 80)
print("SUB-PHASE 3.3.2 COMPLETED")
print("=" * 80)

print(
    "Imputation strategies have been compared descriptively."
)

print(
    "\nImportant:"
    "\n- No final imputation strategy is selected solely from this analysis."
    "\n- No target-derived imputation was performed."
    "\n- No observations were removed from the source analytical dataset."
    "\n- Final model preprocessing will be fitted on training data only."
)