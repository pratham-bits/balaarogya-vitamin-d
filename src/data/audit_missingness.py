from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


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

EDA_DIR = PROJECT_ROOT / "data" / "processed" / "eda"


# ============================================================
# VARIABLES
# ============================================================

TARGET = "LBXVIDMS"
AGE = "RIDAGEYR"
SEX = "RIAGENDR"
WEIGHT = "BMXWT"
MILK = "DBQ197"
DIETARY_VITD = "DR1TVD"


# ============================================================
# HELPERS
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def mann_whitney_comparison(
    df: pd.DataFrame,
    variable: str,
    missing_flag: str,
):
    """
    Compare a numeric variable between:
    - observations where the target variable is observed
    - observations where the target variable is missing
    """

    observed = df.loc[
        ~df[missing_flag],
        variable,
    ].dropna()

    missing = df.loc[
        df[missing_flag],
        variable,
    ].dropna()

    if len(observed) == 0 or len(missing) == 0:
        return {
            "variable": variable,
            "observed_n": len(observed),
            "missing_n": len(missing),
            "observed_mean": float("nan"),
            "missing_mean": float("nan"),
            "p_value": float("nan"),
        }

    statistic, p_value = stats.mannwhitneyu(
        observed,
        missing,
        alternative="two-sided",
    )

    return {
        "variable": variable,
        "observed_n": len(observed),
        "missing_n": len(missing),
        "observed_mean": observed.mean(),
        "missing_mean": missing.mean(),
        "p_value": p_value,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BalAarogya - Vitamin-D Module")
    print("PHASE 3.3 - MISSING DATA & PREPROCESSING")
    print("SUB-PHASE 3.3.1 - MISSINGNESS MECHANISM AUDIT")
    print("=" * 80)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print_section("DATASET")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Create missingness indicators
    # --------------------------------------------------------

    df["BMXWT_missing"] = df[WEIGHT].isna()
    df["DR1TVD_missing"] = df[DIETARY_VITD].isna()

    # --------------------------------------------------------
    # Overall missingness
    # --------------------------------------------------------

    print_section("OVERALL MISSINGNESS")

    missing_summary = pd.DataFrame(
        {
            "Missing_Count": df.isna().sum(),
            "Missing_Percentage": (
                df.isna().mean() * 100
            ),
        }
    )

    print(missing_summary)

    # --------------------------------------------------------
    # 1. DR1TVD missingness
    # --------------------------------------------------------

    print_section(
        "DR1TVD MISSINGNESS"
    )

    dr1tvd_missing_counts = (
        df["DR1TVD_missing"]
        .value_counts()
        .sort_index()
    )

    print(
        dr1tvd_missing_counts
    )

    print(
        f"\nDR1TVD observed: "
        f"{(~df['DR1TVD_missing']).sum()}"
    )

    print(
        f"DR1TVD missing: "
        f"{df['DR1TVD_missing'].sum()}"
    )

    print(
        f"DR1TVD missing percentage: "
        f"{df['DR1TVD_missing'].mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # 2. Compare numeric variables
    # --------------------------------------------------------

    print_section(
        "NUMERIC VARIABLES: OBSERVED VS MISSING DR1TVD"
    )

    numeric_variables = [
        AGE,
        WEIGHT,
        TARGET,
    ]

    comparison_results = []

    for variable in numeric_variables:

        result = mann_whitney_comparison(
            df,
            variable,
            "DR1TVD_missing",
        )

        comparison_results.append(result)

        print(f"\n{variable}")

        print(
            f"Observed: "
            f"N={result['observed_n']}, "
            f"mean={result['observed_mean']:.4f}"
        )

        print(
            f"Missing: "
            f"N={result['missing_n']}, "
            f"mean={result['missing_mean']:.4f}"
        )

        print(
            f"Mann-Whitney p-value: "
            f"{result['p_value']:.6f}"
        )

    comparison_df = pd.DataFrame(
        comparison_results
    )

    comparison_path = (
        EDA_DIR
        / "dr1tvd_missingness_numeric_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    print(
        f"\nSaved: {comparison_path}"
    )

    # --------------------------------------------------------
    # 3. Sex vs DR1TVD missingness
    # --------------------------------------------------------

    print_section(
        "SEX VS DR1TVD MISSINGNESS"
    )

    sex_missing_table = pd.crosstab(
        df["DR1TVD_missing"],
        df[SEX],
    )

    print("\nCounts:")
    print(sex_missing_table)

    sex_missing_pct = pd.crosstab(
        df["DR1TVD_missing"],
        df[SEX],
        normalize="index",
    ) * 100

    print("\nRow percentages:")
    print(sex_missing_pct.round(2))

    chi2, chi_p, dof, expected = (
        stats.chi2_contingency(
            sex_missing_table
        )
    )

    print(
        f"\nChi-square statistic: {chi2:.6f}"
    )

    print(
        f"Chi-square p-value: {chi_p:.6f}"
    )

    # --------------------------------------------------------
    # 4. Milk consumption vs DR1TVD missingness
    # --------------------------------------------------------

    print_section(
        "MILK CONSUMPTION VS DR1TVD MISSINGNESS"
    )

    milk_missing_table = pd.crosstab(
        df["DR1TVD_missing"],
        df[MILK],
    )

    print("\nCounts:")
    print(milk_missing_table)

    milk_missing_pct = pd.crosstab(
        df["DR1TVD_missing"],
        df[MILK],
        normalize="index",
    ) * 100

    print("\nRow percentages:")
    print(milk_missing_pct.round(2))

    # --------------------------------------------------------
    # 5. Missingness association with target
    # --------------------------------------------------------

    print_section(
        "DR1TVD MISSINGNESS VS 25(OH)D"
    )

    observed_target = df.loc[
        ~df["DR1TVD_missing"],
        TARGET,
    ]

    missing_target = df.loc[
        df["DR1TVD_missing"],
        TARGET,
    ]

    print(
        f"Observed DR1TVD target mean: "
        f"{observed_target.mean():.4f}"
    )

    print(
        f"Missing DR1TVD target mean: "
        f"{missing_target.mean():.4f}"
    )

    target_stat, target_p = (
        stats.mannwhitneyu(
            observed_target,
            missing_target,
            alternative="two-sided",
        )
    )

    print(
        f"Mann-Whitney p-value: "
        f"{target_p:.6f}"
    )

    # --------------------------------------------------------
    # 6. Missingness by age
    # --------------------------------------------------------

    print_section(
        "DR1TVD MISSINGNESS BY AGE"
    )

    age_missing = (
        df.groupby(AGE)["DR1TVD_missing"]
        .agg(
            [
                "count",
                "sum",
                "mean",
            ]
        )
    )

    age_missing["missing_percentage"] = (
        age_missing["mean"] * 100
    )

    print(
        age_missing.round(3)
    )

    # --------------------------------------------------------
    # 7. Missingness by milk category
    # --------------------------------------------------------

    print_section(
        "DR1TVD MISSINGNESS BY MILK CATEGORY"
    )

    milk_missing = (
        df.groupby(MILK)["DR1TVD_missing"]
        .agg(
            [
                "count",
                "sum",
                "mean",
            ]
        )
    )

    milk_missing["missing_percentage"] = (
        milk_missing["mean"] * 100
    )

    print(
        milk_missing.round(3)
    )

    # --------------------------------------------------------
    # 8. Missingness indicators relationship
    # --------------------------------------------------------

    print_section(
        "MISSINGNESS INDICATOR CORRELATION"
    )

    missing_indicators = df[
        [
            "BMXWT_missing",
            "DR1TVD_missing",
        ]
    ].astype(int)

    print(
        missing_indicators.corr(
            method="pearson"
        ).round(4)
    )

    # --------------------------------------------------------
    # 9. Missingness visualization
    # --------------------------------------------------------

    print_section(
        "GENERATING MISSINGNESS PLOT"
    )

    missing_rates = pd.Series(
        {
            "BMXWT": df[WEIGHT].isna().mean() * 100,
            "DR1TVD": df[DIETARY_VITD].isna().mean() * 100,
        }
    )

    plt.figure(figsize=(8, 6))

    missing_rates.plot(
        kind="bar",
    )

    plt.ylabel(
        "Missing values (%)"
    )

    plt.xlabel(
        "Feature"
    )

    plt.title(
        "Missingness Rate of Candidate Predictors"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plot_path = (
        EDA_DIR
        / "predictor_missingness_rates.png"
    )

    plt.savefig(
        plot_path,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {plot_path}"
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print_section(
        "SUB-PHASE 3.3.1 COMPLETED"
    )

    print(
        "Missingness mechanism audit completed."
    )

    print(
        "\nImportant:"
        "\n- Missingness patterns are exploratory."
        "\n- Statistical association does not prove a missingness mechanism."
        "\n- No imputation strategy is selected in this sub-phase."
        "\n- No observations are removed in this sub-phase."
    )


if __name__ == "__main__":
    main()