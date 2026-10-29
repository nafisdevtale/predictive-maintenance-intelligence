"""
Statistical Analysis and Hypothesis Testing Engine using SciPy.
Performs Mann-Whitney U tests with Benjamini-Hochberg FDR correction and Chi-Square tests
on permitted predictors without data leakage.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.config import (
    ALL_PREDICTOR_COLUMNS,
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    OUTPUTS_DIR,
    TARGET_COLUMN,
)
from src.data.loader import load_dataset
from src.features.preprocessor import ALL_NUMERIC_FEATURES, FeatureEngineer


def _ensure_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derives engineered features using FeatureEngineer if not already present."""
    missing = [c for c in ENGINEERED_FEATURES if c not in df.columns]
    if missing:
        fe = FeatureEngineer()
        return fe.transform(df)
    return df.copy()


def compute_mann_whitney_tests(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Performs two-sample Mann-Whitney U tests comparing failure vs. non-failure machines
    across numeric features, applying Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing predictors and TARGET_COLUMN ('Machine failure').
    features : list of str, optional
        Numeric features to test. Defaults to ALL_NUMERIC_FEATURES (raw + engineered).
    alpha : float
        Significance threshold for FDR-adjusted p-values (default: 0.05).

    Returns
    -------
    pd.DataFrame
        Detailed statistical report with medians, IQRs, U-stats, raw p-values,
        FDR-adjusted p-values, rank-biserial effect sizes, and significance flags.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' must be present in DataFrame.")

    df_prepared = _ensure_engineered_features(df)
    test_features = features if features is not None else ALL_NUMERIC_FEATURES

    results: List[Dict[str, Any]] = []

    for feat in test_features:
        if feat not in df_prepared.columns:
            raise ValueError(f"Feature '{feat}' not found in DataFrame.")

        # Split into Failure (1) and Non-failure (0) groups
        group_fail = df_prepared.loc[df_prepared[TARGET_COLUMN] == 1, feat].dropna()
        group_norm = df_prepared.loc[df_prepared[TARGET_COLUMN] == 0, feat].dropna()

        n_fail = int(len(group_fail))
        n_norm = int(len(group_norm))

        if n_fail < 3 or n_norm < 3:
            results.append(
                {
                    "feature": feat,
                    "n_non_failure": n_norm,
                    "n_failure": n_fail,
                    "median_non_failure": np.nan,
                    "iqr_non_failure": np.nan,
                    "median_failure": np.nan,
                    "iqr_failure": np.nan,
                    "median_difference": np.nan,
                    "u_statistic": np.nan,
                    "raw_p_value": np.nan,
                    "rank_biserial_r": np.nan,
                    "warning": "Insufficient sample size (minimum 3 required per group).",
                }
            )
            continue

        med_fail = float(group_fail.median())
        med_norm = float(group_norm.median())
        iqr_fail = float(stats.iqr(group_fail))
        iqr_norm = float(stats.iqr(group_norm))
        med_diff = med_fail - med_norm

        # Mann-Whitney U test (two-sided)
        # Passing group_fail first so U corresponds to failure group
        mw_res = stats.mannwhitneyu(group_fail, group_norm, alternative="two-sided")
        u_stat = float(mw_res.statistic)
        p_val = float(mw_res.pvalue)

        # Rank-biserial correlation effect size: r = (2U / (n1 * n2)) - 1
        # Bounded in [-1, +1]. Positive indicates failure group has higher ranks.
        rank_biserial = float((2.0 * u_stat / (n_fail * n_norm)) - 1.0)

        results.append(
            {
                "feature": feat,
                "n_non_failure": n_norm,
                "n_failure": n_fail,
                "median_non_failure": round(med_norm, 4),
                "iqr_non_failure": round(iqr_norm, 4),
                "median_failure": round(med_fail, 4),
                "iqr_failure": round(iqr_fail, 4),
                "median_difference": round(med_diff, 4),
                "u_statistic": round(u_stat, 2),
                "raw_p_value": p_val,
                "rank_biserial_r": round(rank_biserial, 4),
                "warning": None,
            }
        )

    res_df = pd.DataFrame(results)

    # Benjamini-Hochberg (BH) False Discovery Rate (FDR) correction
    valid_pvals = res_df["raw_p_value"].dropna()
    if not valid_pvals.empty:
        # Use scipy.stats.false_discovery_control
        adj_pvals = stats.false_discovery_control(valid_pvals.values, method="bh")
        res_df.loc[valid_pvals.index, "fdr_adjusted_p_value"] = adj_pvals
        res_df["is_significant_fdr"] = res_df["fdr_adjusted_p_value"] < alpha
    else:
        res_df["fdr_adjusted_p_value"] = np.nan
        res_df["is_significant_fdr"] = False

    return res_df


def compute_chi2_type_test(
    df: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Performs Chi-Square test of independence between machine Type (L, M, H)
    and Machine failure status, including Cochran's assumption check and Cramer's V.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing 'Type' and TARGET_COLUMN.
    alpha : float
        Significance threshold (default: 0.05).

    Returns
    -------
    dict
        Structured results including contingency table, test statistics,
        expected cell count checks, and Cramer's V effect size.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    if "Type" not in df.columns or TARGET_COLUMN not in df.columns:
        raise ValueError(f"DataFrame must contain 'Type' and '{TARGET_COLUMN}'.")

    # Contingency Table
    contingency = pd.crosstab(
        df["Type"],
        df[TARGET_COLUMN],
        margins=False,
    )

    # Ensure required shape (at least 2x2)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {
            "status": "ERROR",
            "warning": "Contingency table has fewer than 2 rows or 2 columns.",
            "is_significant": False,
        }

    # Chi-square test
    chi2_res = stats.chi2_contingency(contingency)
    chi2_stat = float(chi2_res.statistic)
    p_val = float(chi2_res.pvalue)
    dof = int(chi2_res.dof)
    expected = chi2_res.expected_freq

    # Cochran's Rule Check:
    # 1. No expected frequency < 1.0
    # 2. No more than 20% of expected frequencies < 5.0
    min_expected = float(expected.min())
    pct_under_5 = float((expected < 5.0).mean() * 100.0)
    assumptions_met = bool(min_expected >= 1.0 and pct_under_5 <= 20.0)

    # Cramer's V Effect Size: sqrt(chi2 / (n * min(r - 1, c - 1)))
    n_total = int(contingency.values.sum())
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = float(np.sqrt(chi2_stat / (n_total * min_dim))) if (n_total * min_dim) > 0 else 0.0

    # Failure rates per Type
    breakdown: Dict[str, Dict[str, Any]] = {}
    for t in contingency.index:
        total_type = int(contingency.loc[t].sum())
        fails = int(contingency.loc[t, 1]) if 1 in contingency.columns else 0
        rate = (fails / total_type * 100.0) if total_type > 0 else 0.0
        breakdown[str(t)] = {
            "total_count": total_type,
            "failure_count": fails,
            "failure_rate_pct": round(rate, 3),
        }

    return {
        "status": "VALID",
        "contingency_table": {
            str(idx): {str(col): int(val) for col, val in row.items()}
            for idx, row in contingency.iterrows()
        },
        "breakdown_by_type": breakdown,
        "chi2_statistic": round(chi2_stat, 4),
        "degrees_of_freedom": dof,
        "p_value": p_val,
        "is_significant": bool(p_val < alpha),
        "cramers_v": round(cramers_v, 4),
        "assumptions": {
            "min_expected_frequency": round(min_expected, 3),
            "pct_expected_under_5": round(pct_under_5, 2),
            "assumptions_met": assumptions_met,
            "warning": (
                None
                if assumptions_met
                else "Warning: Expected frequencies below minimum recommendation; interpret cautiously."
            ),
        },
    }


def run_full_statistical_analysis(
    df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Executes comprehensive statistical hypothesis testing suite, synthesizes
    cautious interpretations, and persists artifacts in outputs/.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input dataset. If None, loads from RAW_DATA_PATH.
    output_dir : Path, optional
        Directory to save CSV and JSON outputs (default: OUTPUTS_DIR).
    alpha : float
        Significance threshold for FDR-adjusted tests.

    Returns
    -------
    dict
        Combined report containing numeric tests, categorical test, and cautious interpretations.
    """
    if df is None:
        df = load_dataset()

    target_dir = Path(output_dir) if output_dir else OUTPUTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Numeric Feature Tests (Mann-Whitney U + FDR)
    numeric_df = compute_mann_whitney_tests(df, alpha=alpha)

    # 2. Categorical Feature Test (Chi-Square + Cramer's V)
    chi2_report = compute_chi2_type_test(df, alpha=alpha)

    # 3. Export CSV files
    numeric_csv_path = target_dir / "statistical_tests_numeric.csv"
    numeric_df.to_csv(numeric_csv_path, index=False)

    contingency_matrix = pd.DataFrame(chi2_report["contingency_table"]).T
    contingency_matrix.columns = [f"Machine_failure_{c}" for c in contingency_matrix.columns]
    contingency_csv_path = target_dir / "contingency_table_type.csv"
    contingency_matrix.to_csv(contingency_csv_path)

    # 4. Formulate Cautious, Evidence-Based Interpretations
    sig_features = numeric_df.loc[numeric_df["is_significant_fdr"], "feature"].tolist()
    largest_effect = (
        numeric_df.sort_values(by="rank_biserial_r", key=abs, ascending=False).iloc[0].to_dict()
    )

    cautious_interpretation = {
        "dataset_nature": (
            "The UCI AI4I 2020 dataset is a synthetic simulation benchmark. "
            "All reported statistical associations reflect underlying mathematical generator functions, "
            "not physical empirical field telemetry."
        ),
        "causality_caveat": (
            "Statistical significance and non-zero rank-biserial correlations indicate distributional "
            "differences across classes, NOT direct causal mechanisms. Confounding and inter-sensor "
            "correlations must be evaluated through multivariate modeling."
        ),
        "fdr_correction_summary": (
            f"Of the {len(numeric_df)} tested numeric features, {len(sig_features)} maintained "
            f"statistical significance after Benjamini-Hochberg FDR correction at alpha={alpha}."
        ),
        "strongest_numeric_association": {
            "feature": largest_effect["feature"],
            "rank_biserial_r": largest_effect["rank_biserial_r"],
            "median_failure": largest_effect["median_failure"],
            "median_non_failure": largest_effect["median_non_failure"],
        },
        "machine_type_association": {
            "chi2_stat": chi2_report["chi2_statistic"],
            "p_value": chi2_report["p_value"],
            "cramers_v": chi2_report["cramers_v"],
            "is_significant": chi2_report["is_significant"],
            "interpretation": (
                f"Machine Type exhibits a Cramer's V of {chi2_report['cramers_v']}, indicating a "
                f"{'statistically significant but weak' if chi2_report['is_significant'] else 'non-significant'} "
                f"association with overall machine failure."
            ),
        },
    }

    combined_report = {
        "sample_size": len(df),
        "alpha_threshold": alpha,
        "multiple_testing_correction": "Benjamini-Hochberg (FDR)",
        "numeric_tests": numeric_df.to_dict(orient="records"),
        "categorical_test": chi2_report,
        "interpretations": cautious_interpretation,
    }

    # 5. Export JSON
    json_path = target_dir / "statistical_analysis_report.json"
    with open(json_path, "w") as f:
        json.dump(combined_report, f, indent=2)

    return combined_report


if __name__ == "__main__":
    report = run_full_statistical_analysis()

    print("\n" + "=" * 80)
    print(" PREDICTIVE MAINTENANCE: STATISTICAL HYPOTHESIS TESTING REPORT")
    print("=" * 80)
    print(f"Total Dataset Observations: {report['sample_size']}")
    print(f"Correction Method: {report['multiple_testing_correction']} (alpha = {report['alpha_threshold']})\n")

    print("--- Numeric Features (Mann-Whitney U Test with FDR Correction) ---")
    num_df = pd.DataFrame(report["numeric_tests"])
    cols_display = [
        "feature",
        "median_non_failure",
        "median_failure",
        "median_difference",
        "rank_biserial_r",
        "raw_p_value",
        "fdr_adjusted_p_value",
        "is_significant_fdr",
    ]
    print(num_df[cols_display].to_string(index=False))

    print("\n--- Categorical Feature: Machine Type vs Failure (Chi-Square Test) ---")
    cat = report["categorical_test"]
    print(f"Chi-Square: {cat['chi2_statistic']} (df = {cat['degrees_of_freedom']}, p = {cat['p_value']:.4e})")
    print(f"Cramer's V Effect Size: {cat['cramers_v']}")
    print(f"Expected Frequencies Check Met: {cat['assumptions']['assumptions_met']}")
    print("Breakdown by Machine Type:")
    for t_name, vals in cat["breakdown_by_type"].items():
        print(f"  Type {t_name}: {vals['failure_count']}/{vals['total_count']} failures ({vals['failure_rate_pct']}%)")

    print("\n--- Cautious Analytical Interpretations ---")
    print(f"* Caveat: {report['interpretations']['dataset_nature']}")
    print(f"* Association != Causation: {report['interpretations']['causality_caveat']}")
    print(f"* Top Association: {report['interpretations']['strongest_numeric_association']['feature']} "
          f"(r_rb = {report['interpretations']['strongest_numeric_association']['rank_biserial_r']})")
    print("=" * 80 + "\n")
