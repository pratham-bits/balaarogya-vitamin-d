from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# BalAarogya - Vitamin-D Module
# Phase 3.1 - 25(OH)D Target EDA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
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
)


def main() -> None:

    print("=" * 80)
    print("BalAarogya - Vitamin-D Module")
    print("PHASE 3.1 - 25(OH)D TARGET EDA")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    target = df["LBXVIDMS"]

    print("\n--- Dataset ---")

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Target missing: "
        f"{target.isna().sum()}"
    )

    # --------------------------------------------------------
    # 2. Descriptive statistics
    # --------------------------------------------------------

    print("\n--- 25(OH)D Descriptive Statistics ---")

    print(
        target.describe()
    )

    # --------------------------------------------------------
    # 3. Detailed quantiles
    # --------------------------------------------------------

    print("\n--- 25(OH)D Quantiles ---")

    quantiles = target.quantile(
        [
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )

    print(quantiles)

    # --------------------------------------------------------
    # 4. Skewness
    # --------------------------------------------------------

    print("\n--- Distribution Shape ---")

    print(
        f"Skewness: "
        f"{target.skew():.4f}"
    )

    print(
        f"Kurtosis: "
        f"{target.kurtosis():.4f}"
    )

    # --------------------------------------------------------
    # 5. IQR-based extreme-value check
    # --------------------------------------------------------

    print("\n--- IQR Extreme-Value Check ---")

    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    print(
        f"Q1: {q1:.2f}"
    )

    print(
        f"Q3: {q3:.2f}"
    )

    print(
        f"IQR: {iqr:.2f}"
    )

    print(
        f"Lower IQR bound: "
        f"{lower_bound:.2f}"
    )

    print(
        f"Upper IQR bound: "
        f"{upper_bound:.2f}"
    )

    lower_extreme = (
        target < lower_bound
    ).sum()

    upper_extreme = (
        target > upper_bound
    ).sum()

    print(
        f"Values below lower bound: "
        f"{lower_extreme}"
    )

    print(
        f"Values above upper bound: "
        f"{upper_extreme}"
    )

    # --------------------------------------------------------
    # 6. Age-wise target statistics
    # --------------------------------------------------------

    print("\n--- 25(OH)D by Age ---")

    age_summary = (
        df.groupby("RIDAGEYR")["LBXVIDMS"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
    )

    print(
        age_summary.to_string()
    )

    # --------------------------------------------------------
    # 7. Sex-wise target statistics
    # --------------------------------------------------------

    print("\n--- 25(OH)D by Sex Code ---")

    sex_summary = (
        df.groupby("RIAGENDR")["LBXVIDMS"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max",
            ]
        )
    )

    print(
        sex_summary.to_string()
    )

    # --------------------------------------------------------
    # 8. Histogram
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        target,
        bins=30,
    )

    plt.xlabel(
        "Total 25(OH)D (nmol/L)"
    )

    plt.ylabel(
        "Number of children"
    )

    plt.title(
        "Distribution of Total 25(OH)D"
    )

    plt.tight_layout()

    histogram_path = (
        OUTPUT_DIR
        / "target_25ohd_distribution.png"
    )

    plt.savefig(
        histogram_path,
        dpi=200,
    )

    plt.close()

    print(
        f"\nHistogram saved to:\n"
        f"{histogram_path}"
    )

    # --------------------------------------------------------
    # 9. Boxplot
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.boxplot(
        target,
        orientation="vertical",
    )

    plt.xlabel(
        "Total 25(OH)D (nmol/L)"
    )

    plt.title(
        "25(OH)D Boxplot"
    )

    plt.tight_layout()

    boxplot_path = (
        OUTPUT_DIR
        / "target_25ohd_boxplot.png"
    )

    plt.savefig(
        boxplot_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Boxplot saved to:\n"
        f"{boxplot_path}"
    )

    # --------------------------------------------------------
    # 10. Age-wise boxplot
    # --------------------------------------------------------

    age_groups = [
        group["LBXVIDMS"].values
        for _, group in df.groupby("RIDAGEYR")
    ]

    age_labels = [
        str(age)
        for age in sorted(
            df["RIDAGEYR"].unique()
        )
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.boxplot(
        age_groups,
        tick_labels=age_labels,
    )

    plt.xlabel(
        "Age (years)"
    )

    plt.ylabel(
        "Total 25(OH)D (nmol/L)"
    )

    plt.title(
        "25(OH)D Distribution by Age"
    )

    plt.tight_layout()

    age_boxplot_path = (
        OUTPUT_DIR
        / "target_25ohd_by_age.png"
    )

    plt.savefig(
        age_boxplot_path,
        dpi=200,
    )

    plt.close()

    print(
        f"Age-wise boxplot saved to:\n"
        f"{age_boxplot_path}"
    )

    # --------------------------------------------------------
    # 11. Final status
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PHASE 3.1 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()