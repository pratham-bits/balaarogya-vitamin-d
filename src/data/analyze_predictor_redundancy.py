from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import (
    mannwhitneyu,
    pearsonr,
    spearmanr,
)
from sklearn.linear_model import LinearRegression


# ============================================================
# BalAarogya - Vitamin-D Module
# PHASE 3.2 - PREDICTOR EDA
# SUB-PHASE 3.2.3
# PREDICTOR REDUNDANCY, CONFOUNDING & FEATURE REPRESENTATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vitamin_d_nhanes_2017_2018.csv"
)

EDA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eda"
)

EDA_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "LBXVIDMS"


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load the analytical dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Analytical dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "SEQN",
        "RIDAGEYR",
        "RIAGENDR",
        "LBXVIDMS",
        "BMXWT",
        "DBQ197",
        "DR1TVD",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    return df


# ============================================================
# AGE-WEIGHT REDUNDANCY
# ============================================================

def analyze_age_weight_relationship(
    df: pd.DataFrame,
) -> None:
    """Analyze the relationship between age and weight."""

    print("\n" + "=" * 80)
    print("AGE ↔ WEIGHT REDUNDANCY")
    print("=" * 80)

    pair = df[
        ["RIDAGEYR", "BMXWT"]
    ].dropna()

    pearson_r, pearson_p = pearsonr(
        pair["RIDAGEYR"],
        pair["BMXWT"],
    )

    spearman_rho, spearman_p = spearmanr(
        pair["RIDAGEYR"],
        pair["BMXWT"],
    )

    print(f"\nPairwise complete N: {len(pair)}")

    print(
        f"Pearson r: {pearson_r:.6f}"
    )
    print(
        f"Pearson p-value: {pearson_p:.6f}"
    )

    print(
        f"Spearman rho: {spearman_rho:.6f}"
    )
    print(
        f"Spearman p-value: {spearman_p:.6f}"
    )

    print("\nWeight by age:")

    summary = (
        df.groupby("RIDAGEYR")["BMXWT"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
    )

    print(summary.round(3))


# ============================================================
# MILK CONSUMPTION BY AGE
# ============================================================

def analyze_milk_by_age(
    df: pd.DataFrame,
) -> None:
    """Examine whether milk frequency varies with age."""

    print("\n" + "=" * 80)
    print("MILK CONSUMPTION BY AGE")
    print("=" * 80)

    cross_tab = pd.crosstab(
        df["RIDAGEYR"],
        df["DBQ197"],
    )

    print("\nCounts:")
    print(cross_tab)

    row_percentages = (
        pd.crosstab(
            df["RIDAGEYR"],
            df["DBQ197"],
            normalize="index",
        )
        * 100
    )

    print("\nRow percentages:")
    print(row_percentages.round(2))

    print("\nMean DBQ197 by age:")

    mean_by_age = (
        df.groupby("RIDAGEYR")["DBQ197"]
        .agg(
            count="count",
            mean="mean",
            median="median",
        )
    )

    print(mean_by_age.round(3))

    rho, p_value = spearmanr(
        df["RIDAGEYR"],
        df["DBQ197"],
    )

    print(
        f"\nAge ↔ DBQ197 Spearman rho: "
        f"{rho:.6f}"
    )

    print(
        f"Age ↔ DBQ197 p-value: "
        f"{p_value:.6f}"
    )


# ============================================================
# DIETARY VITAMIN D BY AGE
# ============================================================

def analyze_dietary_vitamin_d_by_age(
    df: pd.DataFrame,
) -> None:
    """Examine whether dietary Vitamin-D intake varies with age."""

    print("\n" + "=" * 80)
    print("DIETARY VITAMIN D BY AGE")
    print("=" * 80)

    summary = (
        df.groupby("RIDAGEYR")["DR1TVD"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
    )

    print(summary.round(3))

    pair = df[
        ["RIDAGEYR", "DR1TVD"]
    ].dropna()

    pearson_r, pearson_p = pearsonr(
        pair["RIDAGEYR"],
        pair["DR1TVD"],
    )

    spearman_rho, spearman_p = spearmanr(
        pair["RIDAGEYR"],
        pair["DR1TVD"],
    )

    print(
        f"\nPairwise complete N: {len(pair)}"
    )

    print(
        f"Pearson r: {pearson_r:.6f}"
    )

    print(
        f"Pearson p-value: {pearson_p:.6f}"
    )

    print(
        f"Spearman rho: {spearman_rho:.6f}"
    )

    print(
        f"Spearman p-value: {spearman_p:.6f}"
    )


# ============================================================
# DR1TVD MISSINGNESS ANALYSIS
# ============================================================

def analyze_dr1tvd_missingness(
    df: pd.DataFrame,
) -> None:
    """
    Compare children with observed vs missing dietary
    Vitamin-D intake.
    """

    print("\n" + "=" * 80)
    print("DR1TVD MISSINGNESS ANALYSIS")
    print("=" * 80)

    df = df.copy()

    df["DR1TVD_missing"] = (
        df["DR1TVD"].isna()
    )

    missing_count = (
        df["DR1TVD_missing"].sum()
    )

    observed_count = (
        (~df["DR1TVD_missing"]).sum()
    )

    print(
        f"\nObserved DR1TVD: {observed_count}"
    )

    print(
        f"Missing DR1TVD: {missing_count}"
    )

    print(
        f"Missing percentage: "
        f"{missing_count / len(df) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Numeric variables
    # --------------------------------------------------------

    numeric_variables = [
        "RIDAGEYR",
        "BMXWT",
        TARGET,
    ]

    print(
        "\n--- Numeric comparison: "
        "Observed vs Missing DR1TVD ---"
    )

    for variable in numeric_variables:

        observed = (
            df.loc[
                ~df["DR1TVD_missing"],
                variable,
            ]
            .dropna()
        )

        missing = (
            df.loc[
                df["DR1TVD_missing"],
                variable,
            ]
            .dropna()
        )

        print(f"\n{variable}")

        print(
            f"Observed DR1TVD: "
            f"N={len(observed)}, "
            f"mean={observed.mean():.3f}, "
            f"median={observed.median():.3f}"
        )

        print(
            f"Missing DR1TVD: "
            f"N={len(missing)}, "
            f"mean={missing.mean():.3f}, "
            f"median={missing.median():.3f}"
        )

        if len(observed) > 0 and len(missing) > 0:

            statistic, p_value = mannwhitneyu(
                observed,
                missing,
                alternative="two-sided",
            )

            print(
                f"Mann-Whitney U p-value: "
                f"{p_value:.6f}"
            )

    # --------------------------------------------------------
    # Sex comparison
    # --------------------------------------------------------

    print(
        "\n--- Sex distribution by DR1TVD missingness ---"
    )

    sex_table = pd.crosstab(
        df["DR1TVD_missing"],
        df["RIAGENDR"],
    )

    print("\nCounts:")
    print(sex_table)

    sex_percentages = (
        pd.crosstab(
            df["DR1TVD_missing"],
            df["RIAGENDR"],
            normalize="index",
        )
        * 100
    )

    print("\nRow percentages:")
    print(sex_percentages.round(2))


# ============================================================
# SIMPLE VIF DIAGNOSTIC
# ============================================================

def calculate_vif(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate VIF for the numeric predictor set.

    VIF is calculated using complete observations for
    the predictors included here.

    This is a diagnostic, not an automatic feature
    deletion rule.
    """

    columns = [
        "RIDAGEYR",
        "BMXWT",
        "DR1TVD",
    ]

    data = df[columns].dropna()

    results = []

    for target_column in columns:

        other_columns = [
            column
            for column in columns
            if column != target_column
        ]

        X = data[other_columns]
        y = data[target_column]

        model = LinearRegression()

        model.fit(X, y)

        r_squared = model.score(X, y)

        if r_squared >= 1.0:
            vif = float("inf")
        else:
            vif = 1.0 / (1.0 - r_squared)

        results.append(
            {
                "Variable": target_column,
                "VIF": vif,
                "R_squared": r_squared,
            }
        )

    return pd.DataFrame(results)


def print_vif(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print("NUMERIC PREDICTOR MULTICOLLINEARITY DIAGNOSTIC")
    print("=" * 80)

    columns = [
        "RIDAGEYR",
        "BMXWT",
        "DR1TVD",
    ]

    complete_n = (
        df[columns]
        .dropna()
        .shape[0]
    )

    print(
        f"\nComplete-case N for VIF: "
        f"{complete_n}"
    )

    vif_df = calculate_vif(df)

    print(
        "\nVIF results:"
    )

    print(
        vif_df.round(4).to_string(
            index=False
        )
    )


# ============================================================
# VISUALIZATION: AGE vs WEIGHT
# ============================================================

def plot_age_vs_weight(
    df: pd.DataFrame,
) -> None:

    pair = df[
        ["RIDAGEYR", "BMXWT"]
    ].dropna()

    plt.figure(figsize=(9, 6))

    plt.scatter(
        pair["RIDAGEYR"],
        pair["BMXWT"],
        alpha=0.55,
    )

    plt.xlabel("Age (years)")
    plt.ylabel("Weight (kg)")
    plt.title("Age vs Weight")

    plt.xticks(
        sorted(
            pair["RIDAGEYR"].unique()
        )
    )

    plt.tight_layout()

    output_path = (
        EDA_DIR
        / "redundancy_age_vs_weight.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"\nSaved: {output_path}"
    )


# ============================================================
# VISUALIZATION: DIETARY VITAMIN D BY AGE
# ============================================================

def plot_dr1tvd_by_age(
    df: pd.DataFrame,
) -> None:

    grouped_data = []

    age_labels = []

    for age in sorted(
        df["RIDAGEYR"].dropna().unique()
    ):

        values = (
            df.loc[
                df["RIDAGEYR"] == age,
                "DR1TVD",
            ]
            .dropna()
            .values
        )

        if len(values) == 0:
            continue

        grouped_data.append(values)
        age_labels.append(str(int(age)))

    plt.figure(figsize=(9, 6))

    plt.boxplot(
        grouped_data,
        tick_labels=age_labels,
    )

    plt.xlabel("Age (years)")
    plt.ylabel(
        "Dietary Vitamin D (µg)"
    )

    plt.title(
        "Dietary Vitamin D by Age"
    )

    plt.tight_layout()

    output_path = (
        EDA_DIR
        / "redundancy_dr1tvd_by_age.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"\nSaved: {output_path}"
    )


# ============================================================
# VISUALIZATION: MILK BY AGE
# ============================================================

def plot_milk_by_age(
    df: pd.DataFrame,
) -> None:

    grouped_data = []

    age_labels = []

    for age in sorted(
        df["RIDAGEYR"].dropna().unique()
    ):

        values = (
            df.loc[
                df["RIDAGEYR"] == age,
                "DBQ197",
            ]
            .dropna()
            .values
        )

        if len(values) == 0:
            continue

        grouped_data.append(values)
        age_labels.append(str(int(age)))

    plt.figure(figsize=(9, 6))

    plt.boxplot(
        grouped_data,
        tick_labels=age_labels,
    )

    plt.xlabel("Age (years)")
    plt.ylabel(
        "Milk Consumption Frequency Code"
    )

    plt.title(
        "Milk Consumption Frequency by Age"
    )

    plt.tight_layout()

    output_path = (
        EDA_DIR
        / "redundancy_milk_by_age.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"\nSaved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("BalAarogya - Vitamin-D Module")
    print("PHASE 3.2 - PREDICTOR EDA")
    print(
        "SUB-PHASE 3.2.3 - "
        "REDUNDANCY, CONFOUNDING & FEATURE REPRESENTATION"
    )
    print("=" * 80)

    df = load_dataset()

    print("\n--- Dataset ---")
    print(f"Rows: {len(df)}")

    # --------------------------------------------------------
    # Analyses
    # --------------------------------------------------------

    analyze_age_weight_relationship(df)

    analyze_milk_by_age(df)

    analyze_dietary_vitamin_d_by_age(df)

    analyze_dr1tvd_missingness(df)

    print_vif(df)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    print(
        "\n--- Generating Diagnostic Plots ---"
    )

    plot_age_vs_weight(df)

    plot_dr1tvd_by_age(df)

    plot_milk_by_age(df)

    print("\n" + "=" * 80)
    print("SUB-PHASE 3.2.3 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()