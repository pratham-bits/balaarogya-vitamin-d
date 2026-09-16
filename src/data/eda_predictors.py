from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# BalAarogya - Vitamin-D Module
# PHASE 3.2 - PREDICTOR EDA
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


PRIMARY_NUMERIC = [
    "RIDAGEYR",
    "BMXWT",
    "DR1TVD",
]

PRIMARY_CATEGORICAL = [
    "RIAGENDR",
    "DBQ197",
]


def load_dataset() -> pd.DataFrame:
    """Load the analytical Vitamin-D dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Analytical dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


def print_dataset_overview(df: pd.DataFrame) -> None:
    """Print basic dataset information."""

    print("\n--- Dataset Overview ---")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")


def print_numeric_audit(df: pd.DataFrame) -> None:
    """Print descriptive statistics for numeric predictors."""

    print("\n" + "=" * 80)
    print("NUMERIC PREDICTOR AUDIT")
    print("=" * 80)

    for column in PRIMARY_NUMERIC:

        series = df[column]

        print(f"\n--- {column} ---")

        print(f"Valid values: {series.notna().sum()}")
        print(
            f"Missing values: {series.isna().sum()}"
        )
        print(
            f"Missing percentage: "
            f"{series.isna().mean() * 100:.2f}%"
        )

        if series.notna().sum() == 0:
            continue

        print(f"Mean: {series.mean():.4f}")
        print(f"Median: {series.median():.4f}")
        print(f"Std: {series.std():.4f}")
        print(f"Min: {series.min():.4f}")
        print(f"Max: {series.max():.4f}")

        print("\nQuantiles:")
        print(
            series.quantile(
                [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
            )
        )

        print(f"\nSkewness: {series.skew():.4f}")


def print_categorical_audit(df: pd.DataFrame) -> None:
    """Print frequency distributions for categorical predictors."""

    print("\n" + "=" * 80)
    print("CATEGORICAL / ORDINAL PREDICTOR AUDIT")
    print("=" * 80)

    for column in PRIMARY_CATEGORICAL:

        series = df[column]

        print(f"\n--- {column} ---")

        print(f"Valid values: {series.notna().sum()}")
        print(
            f"Missing values: {series.isna().sum()}"
        )
        print(
            f"Missing percentage: "
            f"{series.isna().mean() * 100:.2f}%"
        )

        counts = series.value_counts(
            dropna=False
        ).sort_index()

        percentages = (
            series.value_counts(
                dropna=False,
                normalize=True
            )
            .sort_index()
            * 100
        )

        distribution = pd.DataFrame(
            {
                "Count": counts,
                "Percentage": percentages.round(2),
            }
        )

        print("\nDistribution:")
        print(distribution)


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Save histograms for numeric predictors."""

    for column in PRIMARY_NUMERIC:

        series = df[column].dropna()

        plt.figure(figsize=(10, 6))

        plt.hist(
            series,
            bins=20,
            edgecolor="black",
        )

        plt.xlabel(column)
        plt.ylabel("Number of children")
        plt.title(f"Distribution of {column}")

        plt.tight_layout()

        output_path = (
            EDA_DIR
            / f"predictor_{column.lower()}_distribution.png"
        )

        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"\nSaved: {output_path}")


def plot_categorical_distributions(df: pd.DataFrame) -> None:
    """Save bar charts for categorical predictors."""

    for column in PRIMARY_CATEGORICAL:

        counts = (
            df[column]
            .value_counts(dropna=False)
            .sort_index()
        )

        labels = [
            "Missing" if pd.isna(value) else str(value)
            for value in counts.index
        ]

        plt.figure(figsize=(9, 6))

        plt.bar(
            labels,
            counts.values,
        )

        plt.xlabel(column)
        plt.ylabel("Number of children")
        plt.title(f"Distribution of {column}")

        plt.tight_layout()

        output_path = (
            EDA_DIR
            / f"predictor_{column.lower()}_distribution.png"
        )

        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"\nSaved: {output_path}")


def print_missingness_summary(df: pd.DataFrame) -> None:
    """Print missingness summary for all analytical variables."""

    print("\n" + "=" * 80)
    print("MISSINGNESS SUMMARY")
    print("=" * 80)

    summary = pd.DataFrame(
        {
            "Missing_Count": df.isna().sum(),
            "Missing_Percentage": (
                df.isna().mean() * 100
            ).round(2),
        }
    )

    print(summary)


def main() -> None:

    print("=" * 80)
    print("BalAarogya - Vitamin-D Module")
    print("PHASE 3.2 - PREDICTOR EDA")
    print("SUB-PHASE 3.2.1 - UNIVARIATE PREDICTOR AUDIT")
    print("=" * 80)

    df = load_dataset()

    print_dataset_overview(df)

    print_numeric_audit(df)

    print_categorical_audit(df)

    print_missingness_summary(df)

    print("\n--- Generating Predictor Distribution Plots ---")

    plot_numeric_distributions(df)

    plot_categorical_distributions(df)

    print("\n" + "=" * 80)
    print("SUB-PHASE 3.2.1 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()