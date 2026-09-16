from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# BalAarogya - Vitamin-D Module
# PHASE 3.2 - PREDICTOR EDA
# SUB-PHASE 3.2.2 - PREDICTOR vs 25(OH)D ANALYSIS
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

NUMERIC_PREDICTORS = [
    "RIDAGEYR",
    "BMXWT",
    "DR1TVD",
]

CATEGORICAL_PREDICTORS = [
    "RIAGENDR",
    "DBQ197",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load the analytical Vitamin-D dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Analytical dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise KeyError(
            f"Target column '{TARGET}' not found."
        )

    return df


# ============================================================
# NUMERIC PREDICTOR CORRELATIONS
# ============================================================

def analyze_numeric_relationships(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Pearson and Spearman correlations between
    numeric predictors and the continuous 25(OH)D target.

    Pairwise complete observations are used separately
    for each predictor.
    """

    results = []

    for predictor in NUMERIC_PREDICTORS:

        pair = df[[predictor, TARGET]].dropna()

        x = pair[predictor]
        y = pair[TARGET]

        pearson_r, pearson_p = pearsonr(x, y)
        spearman_rho, spearman_p = spearmanr(x, y)

        results.append(
            {
                "Predictor": predictor,
                "N": len(pair),
                "Pearson_r": pearson_r,
                "Pearson_p": pearson_p,
                "Spearman_rho": spearman_rho,
                "Spearman_p": spearman_p,
            }
        )

    results_df = pd.DataFrame(results)

    return results_df


def print_numeric_relationships(
    results_df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print("NUMERIC PREDICTOR vs 25(OH)D")
    print("=" * 80)

    display_df = results_df.copy()

    for column in [
        "Pearson_r",
        "Pearson_p",
        "Spearman_rho",
        "Spearman_p",
    ]:
        display_df[column] = display_df[column].map(
            lambda value: f"{value:.6f}"
        )

    print(display_df.to_string(index=False))


# ============================================================
# AGE-WISE TARGET ANALYSIS
# ============================================================

def analyze_age_groups(df: pd.DataFrame) -> None:

    print("\n" + "=" * 80)
    print("25(OH)D BY AGE")
    print("=" * 80)

    summary = (
        df.groupby("RIDAGEYR")[TARGET]
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
# SEX-WISE TARGET ANALYSIS
# ============================================================

def analyze_sex_groups(df: pd.DataFrame) -> None:

    print("\n" + "=" * 80)
    print("25(OH)D BY SEX")
    print("=" * 80)

    summary = (
        df.groupby("RIAGENDR")[TARGET]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
    )

    summary["percentage"] = (
        summary["count"] / len(df) * 100
    )

    print(summary.round(3))


# ============================================================
# MILK-FREQUENCY TARGET ANALYSIS
# ============================================================

def analyze_milk_groups(df: pd.DataFrame) -> None:

    print("\n" + "=" * 80)
    print("25(OH)D BY MILK CONSUMPTION FREQUENCY")
    print("=" * 80)

    summary = (
        df.groupby("DBQ197")[TARGET]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
    )

    summary["percentage"] = (
        summary["count"] / len(df) * 100
    )

    print(summary.round(3))


# ============================================================
# CATEGORICAL / ORDINAL ASSOCIATION
# ============================================================

def analyze_categorical_associations(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 80)
    print("CATEGORICAL / ORDINAL ASSOCIATIONS")
    print("=" * 80)

    # Sex is binary.
    sex_pair = df[["RIAGENDR", TARGET]].dropna()

    sex_r, sex_p = pearsonr(
        sex_pair["RIAGENDR"],
        sex_pair[TARGET],
    )

    print("\nRIAGENDR")
    print(f"N: {len(sex_pair)}")
    print(f"Pearson correlation: {sex_r:.6f}")
    print(f"p-value: {sex_p:.6f}")

    # Milk frequency is ordinal.
    milk_pair = df[["DBQ197", TARGET]].dropna()

    milk_rho, milk_p = spearmanr(
        milk_pair["DBQ197"],
        milk_pair[TARGET],
    )

    print("\nDBQ197")
    print(f"N: {len(milk_pair)}")
    print(f"Spearman correlation: {milk_rho:.6f}")
    print(f"p-value: {milk_p:.6f}")


# ============================================================
# SCATTER PLOTS
# ============================================================

def plot_numeric_relationship(
    df: pd.DataFrame,
    predictor: str,
) -> None:

    pair = df[[predictor, TARGET]].dropna()

    plt.figure(figsize=(9, 6))

    plt.scatter(
        pair[predictor],
        pair[TARGET],
        alpha=0.55,
    )

    plt.xlabel(predictor)
    plt.ylabel("Total 25(OH)D (nmol/L)")
    plt.title(
        f"{predictor} vs Total 25(OH)D"
    )

    plt.tight_layout()

    output_path = (
        EDA_DIR
        / f"relationship_{predictor.lower()}_vs_25ohd.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nSaved: {output_path}")


# ============================================================
# BOX PLOTS
# ============================================================

def plot_group_relationship(
    df: pd.DataFrame,
    predictor: str,
) -> None:

    grouped_data = []

    group_labels = []

    for category in sorted(
        df[predictor].dropna().unique()
    ):

        values = (
            df.loc[
                df[predictor] == category,
                TARGET
            ]
            .dropna()
            .values
        )

        if len(values) == 0:
            continue

        grouped_data.append(values)

        if predictor == "RIAGENDR":
            if category == 1:
                label = "1"
            elif category == 2:
                label = "2"
            else:
                label = str(category)

        else:
            label = str(category)

        group_labels.append(label)

    plt.figure(figsize=(10, 6))

    plt.boxplot(
        grouped_data,
        tick_labels=group_labels,
    )

    plt.xlabel(predictor)
    plt.ylabel("Total 25(OH)D (nmol/L)")
    plt.title(
        f"25(OH)D by {predictor}"
    )

    plt.tight_layout()

    output_path = (
        EDA_DIR
        / f"relationship_{predictor.lower()}_vs_25ohd_boxplot.png"
    )

    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nSaved: {output_path}")


# ============================================================
# CORRELATION MATRIX
# ============================================================

def create_correlation_matrix(
    df: pd.DataFrame,
) -> None:

    columns = [
        "RIDAGEYR",
        "BMXWT",
        "DR1TVD",
        TARGET,
    ]

    corr = df[columns].corr(
        method="spearman"
    )

    print("\n" + "=" * 80)
    print("SPEARMAN CORRELATION MATRIX")
    print("=" * 80)

    print(corr.round(4))


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("BalAarogya - Vitamin-D Module")
    print("PHASE 3.2 - PREDICTOR EDA")
    print("SUB-PHASE 3.2.2 - PREDICTOR vs 25(OH)D ANALYSIS")
    print("=" * 80)

    df = load_dataset()

    print("\n--- Dataset ---")
    print(f"Rows: {len(df)}")
    print(f"Target: {TARGET}")
    print(
        f"Target missing: "
        f"{df[TARGET].isna().sum()}"
    )

    # Numeric relationships
    numeric_results = analyze_numeric_relationships(df)

    print_numeric_relationships(
        numeric_results
    )

    # Group analyses
    analyze_age_groups(df)
    analyze_sex_groups(df)
    analyze_milk_groups(df)

    # Categorical / ordinal associations
    analyze_categorical_associations(df)

    # Correlation matrix
    create_correlation_matrix(df)

    # Numeric scatterplots
    print(
        "\n--- Generating Numeric Relationship Plots ---"
    )

    for predictor in NUMERIC_PREDICTORS:
        plot_numeric_relationship(
            df,
            predictor,
        )

    # Group boxplots
    print(
        "\n--- Generating Group Relationship Plots ---"
    )

    for predictor in CATEGORICAL_PREDICTORS:
        plot_group_relationship(
            df,
            predictor,
        )

    print("\n" + "=" * 80)
    print("SUB-PHASE 3.2.2 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()