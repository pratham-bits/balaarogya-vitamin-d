from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

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
    / "outliers"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

TARGET = "LBXVIDMS"

NUMERIC_FEATURES = [
    "RIDAGEYR",
    "BMXWT",
    "DR1TVD",
]


print("=" * 80)
print("BalAarogya - Vitamin-D Module")
print("PHASE 3.3 - MISSING DATA & PREPROCESSING")
print("SUB-PHASE 3.3.3 - OUTLIER & DATA-QUALITY AUDIT")
print("=" * 80)


print("\n" + "=" * 80)
print("DATASET")
print("=" * 80)

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Target: {TARGET}")
print(f"Target missing: {df[TARGET].isna().sum()}")


# ============================================================
# BASIC VALIDITY CHECKS
# ============================================================

print("\n" + "=" * 80)
print("BASIC VALUE VALIDITY CHECK")
print("=" * 80)

for col in NUMERIC_FEATURES + [TARGET]:
    series = df[col]

    print(f"\n{col}")
    print(f"  N: {series.notna().sum()}")
    print(f"  Missing: {series.isna().sum()}")

    if series.notna().any():
        print(f"  Min: {series.min():.4f}")
        print(f"  Max: {series.max():.4f}")

        negative_count = (series.dropna() < 0).sum()
        zero_count = (series.dropna() == 0).sum()

        print(f"  Negative values: {negative_count}")
        print(f"  Zero values: {zero_count}")


# ============================================================
# DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 80)
print("DUPLICATE CHECK")
print("=" * 80)

duplicate_seqn = df["SEQN"].duplicated().sum()
duplicate_rows = df.duplicated().sum()

print(f"Duplicate SEQN values: {duplicate_seqn}")
print(f"Duplicate complete rows: {duplicate_rows}")


# ============================================================
# IQR OUTLIER FUNCTION
# ============================================================

def iqr_outlier_mask(series: pd.Series) -> tuple[pd.Series, float, float]:
    clean = series.dropna()

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (series < lower) | (series > upper)

    return mask, lower, upper


# ============================================================
# OVERALL IQR OUTLIER AUDIT
# ============================================================

print("\n" + "=" * 80)
print("OVERALL IQR OUTLIER AUDIT")
print("=" * 80)

overall_results = []

for col in ["BMXWT", "DR1TVD", TARGET]:

    mask, lower, upper = iqr_outlier_mask(df[col])

    count = mask.sum()
    percentage = (count / len(df)) * 100

    print(f"\n{col}")
    print(f"  Q1: {df[col].quantile(0.25):.4f}")
    print(f"  Q3: {df[col].quantile(0.75):.4f}")
    print(f"  IQR: {(df[col].quantile(0.75) - df[col].quantile(0.25)):.4f}")
    print(f"  Lower bound: {lower:.4f}")
    print(f"  Upper bound: {upper:.4f}")
    print(f"  IQR-flagged observations: {count}")
    print(f"  Percentage: {percentage:.2f}%")

    overall_results.append(
        {
            "Variable": col,
            "IQR_Lower_Bound": lower,
            "IQR_Upper_Bound": upper,
            "IQR_Outlier_Count": count,
            "IQR_Outlier_Percentage": percentage,
        }
    )


overall_results_df = pd.DataFrame(overall_results)

overall_results_df.to_csv(
    OUTPUT_DIR / "overall_iqr_outlier_summary.csv",
    index=False,
)


# ============================================================
# AGE-SPECIFIC OUTLIER AUDIT
# ============================================================

print("\n" + "=" * 80)
print("AGE-SPECIFIC OUTLIER AUDIT")
print("=" * 80)

age_specific_records = []

for age in sorted(df["RIDAGEYR"].dropna().unique()):

    age_df = df[df["RIDAGEYR"] == age]

    print(f"\n--- Age {int(age)} ---")
    print(f"N: {len(age_df)}")

    for col in ["BMXWT", "DR1TVD", TARGET]:

        series = age_df[col].dropna()

        if len(series) < 4:
            print(f"{col}: insufficient observations")
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (age_df[col] < lower) | (age_df[col] > upper)

        count = mask.sum()

        print(
            f"{col}: "
            f"Q1={q1:.3f}, "
            f"Q3={q3:.3f}, "
            f"lower={lower:.3f}, "
            f"upper={upper:.3f}, "
            f"outliers={count}"
        )

        age_specific_records.append(
            {
                "Age": age,
                "Variable": col,
                "N": len(series),
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower_Bound": lower,
                "Upper_Bound": upper,
                "Outlier_Count": count,
                "Outlier_Percentage": (count / len(series)) * 100,
            }
        )


age_specific_df = pd.DataFrame(age_specific_records)

age_specific_df.to_csv(
    OUTPUT_DIR / "age_specific_outlier_summary.csv",
    index=False,
)


# ============================================================
# ROBUST Z-SCORE AUDIT
# ============================================================

print("\n" + "=" * 80)
print("ROBUST Z-SCORE AUDIT")
print("=" * 80)

robust_results = []

for col in ["BMXWT", "DR1TVD", TARGET]:

    series = df[col]

    median = series.median()

    mad = np.median(
        np.abs(series.dropna() - median)
    )

    if mad == 0:
        robust_z = pd.Series(np.nan, index=df.index)
    else:
        robust_z = 0.6745 * (series - median) / mad

    mask = robust_z.abs() > 3.5

    count = mask.sum()

    print(f"\n{col}")
    print(f"  Median: {median:.4f}")
    print(f"  MAD: {mad:.4f}")
    print(f"  Robust |z| > 3.5: {count}")

    robust_results.append(
        {
            "Variable": col,
            "Median": median,
            "MAD": mad,
            "Robust_Z_Threshold": 3.5,
            "Robust_Outlier_Count": count,
        }
    )


robust_results_df = pd.DataFrame(robust_results)

robust_results_df.to_csv(
    OUTPUT_DIR / "robust_zscore_summary.csv",
    index=False,
)


# ============================================================
# EXTREME OBSERVATION TABLE
# ============================================================

print("\n" + "=" * 80)
print("EXTREME OBSERVATION REVIEW")
print("=" * 80)


def print_extremes(column: str, n: int = 10):

    print(f"\nTop {n} highest values: {column}")

    cols = [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "BMXWT",
        "DR1TVD",
        "DBQ197",
        TARGET,
    ]

    available_cols = [c for c in cols if c in df.columns]

    result = (
        df[available_cols]
        .sort_values(column, ascending=False)
        .head(n)
    )

    print(result.to_string(index=False))


print_extremes("BMXWT")
print_extremes("DR1TVD")
print_extremes(TARGET)


# ============================================================
# ZERO DIETARY VITAMIN D REVIEW
# ============================================================

print("\n" + "=" * 80)
print("ZERO DIETARY VITAMIN D REVIEW")
print("=" * 80)

zero_dr1tvd = df[df["DR1TVD"] == 0]

print(f"DR1TVD == 0 observations: {len(zero_dr1tvd)}")

if len(zero_dr1tvd) > 0:
    print("\nAge distribution:")
    print(
        zero_dr1tvd["RIDAGEYR"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTarget summary:")
    print(
        zero_dr1tvd[TARGET]
        .describe()
        .round(3)
        .to_string()
    )


# ============================================================
# NEGATIVE VALUE REVIEW
# ============================================================

print("\n" + "=" * 80)
print("NEGATIVE VALUE REVIEW")
print("=" * 80)

for col in ["BMXWT", "DR1TVD", TARGET]:

    negative_rows = df[df[col] < 0]

    print(
        f"{col}: "
        f"{len(negative_rows)} negative observations"
    )


# ============================================================
# TARGET EXTREME REVIEW
# ============================================================

print("\n" + "=" * 80)
print("TARGET EXTREME REVIEW")
print("=" * 80)

target_mask, target_lower, target_upper = iqr_outlier_mask(df[TARGET])

target_extremes = df.loc[
    target_mask,
    [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "BMXWT",
        "DR1TVD",
        TARGET,
    ],
].sort_values(TARGET)

print(
    f"Target IQR-flagged observations: "
    f"{len(target_extremes)}"
)

if len(target_extremes) > 0:
    print("\nFlagged target observations:")
    print(target_extremes.to_string(index=False))

target_extremes.to_csv(
    OUTPUT_DIR / "target_iqr_flagged_observations.csv",
    index=False,
)


# ============================================================
# OUTLIER FLAG DATASET
# ============================================================

flagged_df = df[
    [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "BMXWT",
        "DR1TVD",
        "DBQ197",
        TARGET,
    ]
].copy()

for col in ["BMXWT", "DR1TVD", TARGET]:

    mask, _, _ = iqr_outlier_mask(df[col])

    flagged_df[f"{col}_iqr_flag"] = mask.astype(int)


flagged_df.to_csv(
    OUTPUT_DIR / "outlier_flags.csv",
    index=False,
)


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

print("\n" + "=" * 80)
print("GENERATING OUTLIER DIAGNOSTIC PLOTS")
print("=" * 80)


# Weight vs age
plt.figure(figsize=(10, 7))

plt.scatter(
    df["RIDAGEYR"],
    df["BMXWT"],
    alpha=0.6,
)

plt.xlabel("Age (years)")
plt.ylabel("Weight (kg)")
plt.title("Weight vs Age — Outlier Review")

plt.tight_layout()

weight_plot = OUTPUT_DIR / "weight_vs_age_outlier_review.png"
plt.savefig(weight_plot, dpi=150)
plt.close()

print(f"Saved: {weight_plot}")


# Dietary vitamin D
plt.figure(figsize=(10, 7))

plt.scatter(
    df["RIDAGEYR"],
    df["DR1TVD"],
    alpha=0.6,
)

plt.xlabel("Age (years)")
plt.ylabel("Dietary Vitamin D (µg)")
plt.title("Dietary Vitamin D vs Age — Outlier Review")

plt.tight_layout()

diet_plot = OUTPUT_DIR / "dr1tvd_vs_age_outlier_review.png"
plt.savefig(diet_plot, dpi=150)
plt.close()

print(f"Saved: {diet_plot}")


# Target
plt.figure(figsize=(10, 7))

plt.scatter(
    df["RIDAGEYR"],
    df[TARGET],
    alpha=0.6,
)

plt.xlabel("Age (years)")
plt.ylabel("Total 25(OH)D (nmol/L)")
plt.title("25(OH)D vs Age — Extreme Value Review")

plt.tight_layout()

target_plot = OUTPUT_DIR / "25ohd_vs_age_outlier_review.png"
plt.savefig(target_plot, dpi=150)
plt.close()

print(f"Saved: {target_plot}")


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

assert len(df) == 702
assert df["SEQN"].is_unique
assert df[TARGET].notna().all()

print("Dataset row count: PASS")
print("SEQN uniqueness: PASS")
print("Target completeness: PASS")

print("\nNo observations were deleted.")
print("No values were modified.")
print("Outlier flags are diagnostic only.")


print("\n" + "=" * 80)
print("SUB-PHASE 3.3.3 COMPLETED")
print("=" * 80)

print(
    "Outlier and data-quality audit completed.\n"
    "No observations have been removed or modified."
)