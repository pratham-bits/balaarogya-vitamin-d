from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
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

TARGET = "LBXVIDMS"
AGE = "RIDAGEYR"
WEIGHT = "BMXWT"
DIETARY_VITD = "DR1TVD"
MILK = "DBQ197"


# ============================================================
# HELPERS
# ============================================================

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def residualize(y: pd.Series, covariates: pd.DataFrame):
    """
    Remove the linear effect of covariates from y.

    Returns residuals and the fitted values.
    """
    y_array = y.to_numpy(dtype=float)
    x_array = covariates.to_numpy(dtype=float)

    # Add intercept
    x_design = np.column_stack(
        [np.ones(len(x_array)), x_array]
    )

    beta, *_ = np.linalg.lstsq(x_design, y_array, rcond=None)

    fitted = x_design @ beta
    residuals = y_array - fitted

    return residuals, fitted


def age_adjusted_partial_correlation(
    df: pd.DataFrame,
    predictor: str,
    target: str,
    age: str,
):
    """
    Compute an age-adjusted partial correlation.

    Both predictor and target are residualized against age,
    then Pearson and Spearman correlations are calculated
    between the residuals.
    """

    subset = df[[predictor, target, age]].dropna().copy()

    x_resid, _ = residualize(
        subset[predictor],
        subset[[age]],
    )

    y_resid, _ = residualize(
        subset[target],
        subset[[age]],
    )

    pearson_r, pearson_p = stats.pearsonr(
        x_resid,
        y_resid,
    )

    spearman_rho, spearman_p = stats.spearmanr(
        x_resid,
        y_resid,
    )

    return {
        "predictor": predictor,
        "N": len(subset),
        "Pearson_partial_r": pearson_r,
        "Pearson_p": pearson_p,
        "Spearman_partial_rho": spearman_rho,
        "Spearman_p": spearman_p,
    }


def regression_r2(
    df: pd.DataFrame,
    predictors: list[str],
    target: str,
):
    """
    Fit ordinary least-squares regression manually and
    return R² and residual sum of squares.

    This avoids requiring statsmodels for this diagnostic.
    """

    subset = df[predictors + [target]].dropna().copy()

    y = subset[target].to_numpy(dtype=float)

    x_values = subset[predictors].to_numpy(dtype=float)

    x_design = np.column_stack(
        [np.ones(len(x_values)), x_values]
    )

    beta, *_ = np.linalg.lstsq(
        x_design,
        y,
        rcond=None,
    )

    predictions = x_design @ beta

    residual_ss = np.sum(
        (y - predictions) ** 2
    )

    total_ss = np.sum(
        (y - np.mean(y)) ** 2
    )

    r_squared = (
        1 - residual_ss / total_ss
        if total_ss > 0
        else np.nan
    )

    return len(subset), r_squared, residual_ss


def nested_f_test(
    df: pd.DataFrame,
    target: str,
    reduced_predictors: list[str],
    full_predictors: list[str],
):
    """
    Compare nested OLS models using an F-test.

    Reduced model must be nested inside full model.
    """

    required = list(
        dict.fromkeys(
            reduced_predictors
            + full_predictors
            + [target]
        )
    )

    subset = df[required].dropna().copy()

    y = subset[target].to_numpy(dtype=float)

    def fit_rss(predictors):
        x = subset[predictors].to_numpy(dtype=float)

        x_design = np.column_stack(
            [np.ones(len(x)), x]
        )

        beta, *_ = np.linalg.lstsq(
            x_design,
            y,
            rcond=None,
        )

        predictions = x_design @ beta

        rss = np.sum(
            (y - predictions) ** 2
        )

        return rss, x_design.shape[1]

    rss_reduced, df_reduced_params = fit_rss(
        reduced_predictors
    )

    rss_full, df_full_params = fit_rss(
        full_predictors
    )

    n = len(subset)

    numerator_df = (
        df_full_params - df_reduced_params
    )

    denominator_df = (
        n - df_full_params
    )

    if numerator_df <= 0 or denominator_df <= 0:
        return {
            "N": n,
            "F": np.nan,
            "p": np.nan,
            "delta_R2": np.nan,
        }

    f_stat = (
        (rss_reduced - rss_full)
        / numerator_df
    ) / (
        rss_full
        / denominator_df
    )

    p_value = stats.f.sf(
        f_stat,
        numerator_df,
        denominator_df,
    )

    _, r2_reduced, _ = regression_r2(
        subset,
        reduced_predictors,
        target,
    )

    _, r2_full, _ = regression_r2(
        subset,
        full_predictors,
        target,
    )

    return {
        "N": n,
        "F": f_stat,
        "p": p_value,
        "delta_R2": r2_full - r2_reduced,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BalAarogya - Vitamin-D Module")
    print("PHASE 3.2 - PREDICTOR EDA")
    print("SUB-PHASE 3.2.4 - ADJUSTED RELATIONSHIPS & NONLINEARITY")
    print("=" * 80)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print_section("DATASET")

    print(f"Rows: {len(df)}")
    print(f"Target: {TARGET}")
    print(f"Target missing: {df[TARGET].isna().sum()}")

    # --------------------------------------------------------
    # 1. AGE-ADJUSTED CONTINUOUS PREDICTORS
    # --------------------------------------------------------

    print_section(
        "AGE-ADJUSTED CONTINUOUS PREDICTOR ASSOCIATIONS"
    )

    continuous_predictors = [
        WEIGHT,
        DIETARY_VITD,
    ]

    adjusted_results = []

    for predictor in continuous_predictors:

        result = age_adjusted_partial_correlation(
            df=df,
            predictor=predictor,
            target=TARGET,
            age=AGE,
        )

        adjusted_results.append(result)

        print(f"\n{predictor}")
        print(f"N: {result['N']}")
        print(
            f"Age-adjusted Pearson r: "
            f"{result['Pearson_partial_r']:.6f}"
        )
        print(
            f"Pearson p-value: "
            f"{result['Pearson_p']:.6f}"
        )
        print(
            f"Age-adjusted Spearman rho: "
            f"{result['Spearman_partial_rho']:.6f}"
        )
        print(
            f"Spearman p-value: "
            f"{result['Spearman_p']:.6f}"
        )

    adjusted_results_df = pd.DataFrame(
        adjusted_results
    )

    adjusted_results_path = (
        EDA_DIR
        / "adjusted_continuous_associations.csv"
    )

    adjusted_results_df.to_csv(
        adjusted_results_path,
        index=False,
    )

    print(
        f"\nSaved: {adjusted_results_path}"
    )

    # --------------------------------------------------------
    # 2. AGE-ADJUSTED MILK ASSOCIATION
    # --------------------------------------------------------

    print_section(
        "AGE-ADJUSTED MILK CONSUMPTION ASSOCIATION"
    )

    milk_result = age_adjusted_partial_correlation(
        df=df,
        predictor=MILK,
        target=TARGET,
        age=AGE,
    )

    print(f"N: {milk_result['N']}")
    print(
        f"Age-adjusted Pearson r: "
        f"{milk_result['Pearson_partial_r']:.6f}"
    )
    print(
        f"Pearson p-value: "
        f"{milk_result['Pearson_p']:.6f}"
    )
    print(
        f"Age-adjusted Spearman rho: "
        f"{milk_result['Spearman_partial_rho']:.6f}"
    )
    print(
        f"Spearman p-value: "
        f"{milk_result['Spearman_p']:.6f}"
    )

    # --------------------------------------------------------
    # 3. AGE-ADJUSTED MILK: LINEAR VS CATEGORICAL
    # --------------------------------------------------------

    print_section(
        "MILK FREQUENCY: ORDINAL VS CATEGORICAL REPRESENTATION"
    )

    # Because DBQ197 = 4 contains only one observation,
    # category 4 is excluded from this diagnostic.
    milk_df = df[
        df[MILK].isin([0, 1, 2, 3])
    ].copy()

    print(
        "Category 4 is excluded from this diagnostic "
        "because it contains only one observation."
    )

    # Reduced model: age only
    # Full model: age + ordinal milk
    ordinal_test = nested_f_test(
        df=milk_df,
        target=TARGET,
        reduced_predictors=[AGE],
        full_predictors=[AGE, MILK],
    )

    print("\nOrdinal milk model:")
    print(f"N: {ordinal_test['N']}")
    print(
        f"F-statistic: "
        f"{ordinal_test['F']:.6f}"
    )
    print(
        f"p-value: "
        f"{ordinal_test['p']:.6f}"
    )
    print(
        f"Incremental R²: "
        f"{ordinal_test['delta_R2']:.6f}"
    )

    # --------------------------------------------------------
    # 4. AGE NONLINEARITY
    # --------------------------------------------------------

    print_section(
        "AGE RELATIONSHIP: LINEAR VS QUADRATIC"
    )

    age_linear = df[
        [AGE, TARGET]
    ].dropna().copy()

    age_linear_r2 = regression_r2(
        age_linear,
        predictors=[AGE],
        target=TARGET,
    )

    age_linear_r2_value = age_linear_r2[1]

    age_linear["AGE_SQUARED"] = (
        age_linear[AGE] ** 2
    )

    age_quadratic_r2 = regression_r2(
        age_linear,
        predictors=[AGE, "AGE_SQUARED"],
        target=TARGET,
    )

    age_quadratic_r2_value = age_quadratic_r2[1]

    print(f"N: {age_linear_r2[0]}")
    print(
        f"Linear age R²: "
        f"{age_linear_r2_value:.6f}"
    )
    print(
        f"Quadratic age R²: "
        f"{age_quadratic_r2_value:.6f}"
    )
    print(
        f"ΔR²: "
        f"{age_quadratic_r2_value - age_linear_r2_value:.6f}"
    )

    # --------------------------------------------------------
    # 5. WEIGHT NONLINEARITY AFTER AGE ADJUSTMENT
    # --------------------------------------------------------

    print_section(
        "WEIGHT RELATIONSHIP: LINEAR VS QUADRATIC AFTER AGE ADJUSTMENT"
    )

    weight_df = df[
        [AGE, WEIGHT, TARGET]
    ].dropna().copy()

    weight_df["WEIGHT_SQUARED"] = (
        weight_df[WEIGHT] ** 2
    )

    weight_nonlinear_test = nested_f_test(
        df=weight_df,
        target=TARGET,
        reduced_predictors=[AGE, WEIGHT],
        full_predictors=[
            AGE,
            WEIGHT,
            "WEIGHT_SQUARED",
        ],
    )

    print(f"N: {weight_nonlinear_test['N']}")
    print(
        f"F-statistic: "
        f"{weight_nonlinear_test['F']:.6f}"
    )
    print(
        f"p-value: "
        f"{weight_nonlinear_test['p']:.6f}"
    )
    print(
        f"Incremental R²: "
        f"{weight_nonlinear_test['delta_R2']:.6f}"
    )

    # --------------------------------------------------------
    # 6. DIETARY VITAMIN-D NONLINEARITY AFTER AGE ADJUSTMENT
    # --------------------------------------------------------

    print_section(
        "DIETARY VITAMIN D: LINEAR VS QUADRATIC AFTER AGE ADJUSTMENT"
    )

    dietary_df = df[
        [AGE, DIETARY_VITD, TARGET]
    ].dropna().copy()

    dietary_df["DR1TVD_SQUARED"] = (
        dietary_df[DIETARY_VITD] ** 2
    )

    dietary_nonlinear_test = nested_f_test(
        df=dietary_df,
        target=TARGET,
        reduced_predictors=[
            AGE,
            DIETARY_VITD,
        ],
        full_predictors=[
            AGE,
            DIETARY_VITD,
            "DR1TVD_SQUARED",
        ],
    )

    print(f"N: {dietary_nonlinear_test['N']}")
    print(
        f"F-statistic: "
        f"{dietary_nonlinear_test['F']:.6f}"
    )
    print(
        f"p-value: "
        f"{dietary_nonlinear_test['p']:.6f}"
    )
    print(
        f"Incremental R²: "
        f"{dietary_nonlinear_test['delta_R2']:.6f}"
    )

    # --------------------------------------------------------
    # 7. AGE-ADJUSTED MILK GROUP SUMMARY
    # --------------------------------------------------------

    print_section(
        "AGE-ADJUSTED MILK GROUP SUMMARY"
    )

    milk_complete = milk_df[
        [AGE, MILK, TARGET]
    ].dropna().copy()

    target_residuals, _ = residualize(
        milk_complete[TARGET],
        milk_complete[[AGE]],
    )

    milk_complete["Age_Adjusted_25OHD"] = (
        target_residuals
    )

    adjusted_milk_summary = (
        milk_complete
        .groupby(MILK)["Age_Adjusted_25OHD"]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
            ]
        )
    )

    print(adjusted_milk_summary)

    adjusted_milk_path = (
        EDA_DIR
        / "adjusted_milk_summary.csv"
    )

    adjusted_milk_summary.to_csv(
        adjusted_milk_path
    )

    print(
        f"\nSaved: {adjusted_milk_path}"
    )

    # --------------------------------------------------------
    # 8. DIAGNOSTIC PLOTS
    # --------------------------------------------------------

    print_section(
        "GENERATING ADJUSTED RELATIONSHIP PLOTS"
    )

    # Age-adjusted weight relationship
    weight_complete = df[
        [AGE, WEIGHT, TARGET]
    ].dropna().copy()

    weight_x_resid, _ = residualize(
        weight_complete[WEIGHT],
        weight_complete[[AGE]],
    )

    weight_y_resid, _ = residualize(
        weight_complete[TARGET],
        weight_complete[[AGE]],
    )

    plt.figure(figsize=(10, 7))
    plt.scatter(
        weight_x_resid,
        weight_y_resid,
        alpha=0.55,
    )
    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )
    plt.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )
    plt.xlabel(
        "Age-adjusted Weight"
    )
    plt.ylabel(
        "Age-adjusted 25(OH)D"
    )
    plt.title(
        "Age-adjusted Weight vs 25(OH)D"
    )
    plt.tight_layout()

    weight_plot_path = (
        EDA_DIR
        / "adjusted_weight_vs_25ohd.png"
    )

    plt.savefig(
        weight_plot_path,
        dpi=150,
    )
    plt.close()

    print(
        f"Saved: {weight_plot_path}"
    )

    # Age-adjusted dietary Vitamin D relationship
    dietary_complete = df[
        [AGE, DIETARY_VITD, TARGET]
    ].dropna().copy()

    dietary_x_resid, _ = residualize(
        dietary_complete[DIETARY_VITD],
        dietary_complete[[AGE]],
    )

    dietary_y_resid, _ = residualize(
        dietary_complete[TARGET],
        dietary_complete[[AGE]],
    )

    plt.figure(figsize=(10, 7))
    plt.scatter(
        dietary_x_resid,
        dietary_y_resid,
        alpha=0.55,
    )
    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )
    plt.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )
    plt.xlabel(
        "Age-adjusted Dietary Vitamin D"
    )
    plt.ylabel(
        "Age-adjusted 25(OH)D"
    )
    plt.title(
        "Age-adjusted Dietary Vitamin D vs 25(OH)D"
    )
    plt.tight_layout()

    dietary_plot_path = (
        EDA_DIR
        / "adjusted_dr1tvd_vs_25ohd.png"
    )

    plt.savefig(
        dietary_plot_path,
        dpi=150,
    )
    plt.close()

    print(
        f"Saved: {dietary_plot_path}"
    )

    # --------------------------------------------------------
    # 9. FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "PHASE 3.2.4 COMPLETED"
    )

    print(
        "Adjusted association and nonlinearity diagnostics "
        "have been generated."
    )

    print(
        "\nImportant:"
        "\n- These analyses are exploratory."
        "\n- They do not establish causality."
        "\n- No clinical Vitamin-D threshold is introduced."
        "\n- No predictor is permanently accepted/rejected solely "
        "from p-values."
    )


if __name__ == "__main__":
    main()