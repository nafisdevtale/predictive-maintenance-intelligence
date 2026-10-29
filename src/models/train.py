"""
Model training, cross-validation, threshold optimization, and artifact serialization.
Ensures complete encapsulation: every saved Pipeline contains both preprocessor and classifier.
Strictly prevents leakage: X_test/y_test are never used in model selection or threshold tuning.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_PREDICTOR_COLUMNS,
    LOGISTIC_REGRESSION_PIPELINE_PATH,
    METADATA_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_FOREST_PIPELINE_PATH,
    RANDOM_STATE,
    SAFETY_DISCLAIMER,
)
from src.data.loader import load_dataset
from src.features.preprocessor import (
    ALL_NUMERIC_FEATURES,
    create_preprocessor,
    get_train_test_split,
)
from src.models.evaluate import (
    calculate_classification_metrics,
    find_optimal_threshold,
    plot_confusion_matrix,
    plot_model_comparison_bar,
    plot_precision_recall_curves,
    plot_roc_curves,
)


def build_logistic_regression_pipeline() -> Pipeline:
    """
    Constructs a complete pipeline combining FeatureEngineer, ColumnTransformer,
    and a balanced Logistic Regression classifier.
    """
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """
    Constructs a complete pipeline combining FeatureEngineer, ColumnTransformer,
    and a regularized Random Forest classifier with balanced subsampling.
    """
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def get_feature_names_from_fitted_pipeline(pipeline: Pipeline) -> List[str]:
    """Extracts output feature names from the fitted preprocessor inside a pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    col_transformer = preprocessor.named_steps["column_transformer"]
    cat_encoder = col_transformer.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(["Type"]))
    return ALL_NUMERIC_FEATURES + cat_names


def perform_oof_cross_validation(
    pipeline_builder,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> np.ndarray:
    """
    Performs Stratified 5-Fold Cross-Validation strictly on training data
    to generate out-of-fold predicted probabilities for class 1.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(X_train), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        fold_pipeline = pipeline_builder()
        fold_pipeline.fit(X_fold_train, y_fold_train)
        probs = fold_pipeline.predict_proba(X_fold_val)[:, 1]
        oof_probs[val_idx] = probs

    return oof_probs


def train_and_evaluate_models() -> Dict[str, Any]:
    """
    Main training workflow:
    1. Stratified split (80/20). X_test/y_test are held out untouched.
    2. 5-Fold Stratified Cross-Validation on X_train to compute out-of-fold probabilities.
    3. Model selection & threshold optimization on X_train only (F2-score).
    4. Fit full pipelines on X_train.
    5. Final unbiased evaluation on X_test/y_test.
    6. Save models, metrics, and visualization artifacts.
    """
    start_time = time.time()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading dataset and performing reproducible stratified 80/20 split...")
    df = load_dataset()
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    baseline_failure_rate = float(y_train.mean())

    print(f"      Train instances: {len(X_train)} (Failures: {int(y_train.sum())}, {baseline_failure_rate * 100:.2f}%)")
    print(f"      Test instances:  {len(X_test)} (Failures: {int(y_test.sum())}, {float(y_test.mean()) * 100:.2f}%)")
    print("      [RULE VERIFIED] X_test/y_test strictly quarantined from training, CV, and threshold tuning.")

    # -------------------------------------------------------------
    # 2. Out-of-fold Cross-Validation on Training Data
    # -------------------------------------------------------------
    print("\n[2/6] Running Stratified 5-Fold Out-of-Fold Cross-Validation on X_train...")
    print("      - Training Logistic Regression Baseline (class_weight='balanced')...")
    oof_probs_lr = perform_oof_cross_validation(build_logistic_regression_pipeline, X_train, y_train)

    print("      - Training Random Forest Classifier (class_weight='balanced_subsample')...")
    oof_probs_rf = perform_oof_cross_validation(build_random_forest_pipeline, X_train, y_train)

    # -------------------------------------------------------------
    # 3. Training-side Evaluation & Threshold Optimization (X_train ONLY)
    # -------------------------------------------------------------
    print("\n[3/6] Evaluating Out-of-Fold Performance & Selecting Decision Threshold...")
    oof_metrics_lr_default = calculate_classification_metrics(y_train, oof_probs_lr, threshold=0.5)
    oof_metrics_rf_default = calculate_classification_metrics(y_train, oof_probs_rf, threshold=0.5)

    thresh_opt_lr = find_optimal_threshold(y_train, oof_probs_lr, metric="f2")
    thresh_opt_rf = find_optimal_threshold(y_train, oof_probs_rf, metric="f2")

    oof_opt_lr = thresh_opt_lr["optimal_threshold"]
    oof_opt_rf = thresh_opt_rf["optimal_threshold"]

    oof_metrics_lr_opt = calculate_classification_metrics(y_train, oof_probs_lr, threshold=oof_opt_lr)
    oof_metrics_rf_opt = calculate_classification_metrics(y_train, oof_probs_rf, threshold=oof_opt_rf)

    print(f"      Logistic Regression OOF Average Precision (PR-AUC): {oof_metrics_lr_default['average_precision_pr_auc']:.4f}")
    print(f"      Random Forest OOF Average Precision (PR-AUC):       {oof_metrics_rf_default['average_precision_pr_auc']:.4f}")
    print(f"      Random Forest F2-optimal threshold on train OOF:   {oof_opt_rf:.2f} (F2 = {oof_metrics_rf_opt['f2_score']:.4f})")

    # Primary Model Selection (Based STRICTLY on Training-side OOF PR-AUC)
    if oof_metrics_rf_default["average_precision_pr_auc"] >= oof_metrics_lr_default["average_precision_pr_auc"]:
        selected_model_name = "Random Forest"
        selected_threshold = oof_opt_rf
        selected_pipeline_builder = build_random_forest_pipeline
    else:
        selected_model_name = "Logistic Regression"
        selected_threshold = oof_opt_lr
        selected_pipeline_builder = build_logistic_regression_pipeline

    print(f"\n      >>> Selected Primary Model: {selected_model_name} (Selected Threshold: {selected_threshold:.2f}) <<<")

    # -------------------------------------------------------------
    # 4. Final Pipeline Fitting on Full X_train
    # -------------------------------------------------------------
    print("\n[4/6] Fitting complete production pipelines on full X_train...")
    lr_pipeline = build_logistic_regression_pipeline()
    lr_pipeline.fit(X_train, y_train)

    rf_pipeline = build_random_forest_pipeline()
    rf_pipeline.fit(X_train, y_train)

    # -------------------------------------------------------------
    # 5. Final Unbiased Evaluation on Held-Out Test Set (X_test / y_test)
    # -------------------------------------------------------------
    print("\n[5/6] Performing final unbiased evaluation on held-out test set (X_test, 2,000 samples)...")
    test_probs_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    test_probs_rf = rf_pipeline.predict_proba(X_test)[:, 1]

    # Evaluate at both default (0.5) and the training-selected threshold
    test_metrics_lr_default = calculate_classification_metrics(y_test, test_probs_lr, threshold=0.5)
    test_metrics_lr_opt = calculate_classification_metrics(y_test, test_probs_lr, threshold=oof_opt_lr)

    test_metrics_rf_default = calculate_classification_metrics(y_test, test_probs_rf, threshold=0.5)
    test_metrics_rf_opt = calculate_classification_metrics(y_test, test_probs_rf, threshold=oof_opt_rf)

    # Feature Importances / Coefficients
    feature_names = get_feature_names_from_fitted_pipeline(rf_pipeline)
    rf_importances = {
        name: round(float(imp), 4)
        for name, imp in zip(feature_names, rf_pipeline.named_steps["classifier"].feature_importances_)
    }
    rf_importances_sorted = dict(sorted(rf_importances.items(), key=lambda item: item[1], reverse=True))

    lr_coefs = {
        name: round(float(coef), 4)
        for name, coef in zip(feature_names, lr_pipeline.named_steps["classifier"].coef_[0])
    }
    lr_coefs_sorted = dict(sorted(lr_coefs.items(), key=lambda item: abs(item[1]), reverse=True))

    # -------------------------------------------------------------
    # 6. Artifact Serialization & Figure Generation
    # -------------------------------------------------------------
    print("\n[6/6] Serializing pipelines and saving publication figures under outputs/figures/...")
    # Serialize both pipelines
    joblib.dump(lr_pipeline, LOGISTIC_REGRESSION_PIPELINE_PATH)
    joblib.dump(rf_pipeline, RANDOM_FOREST_PIPELINE_PATH)
    print(f"      Saved: {LOGISTIC_REGRESSION_PIPELINE_PATH}")
    print(f"      Saved: {RANDOM_FOREST_PIPELINE_PATH}")

    # Generate Figures
    # 1. Confusion Matrices on Test Set
    plot_confusion_matrix(
        test_metrics_rf_default["confusion_matrix"],
        title=f"Random Forest (Test Set, Default Cutoff = 0.50)",
        output_path=figures_dir / "confusion_matrix_rf_test_default.png",
    )
    plot_confusion_matrix(
        test_metrics_rf_opt["confusion_matrix"],
        title=f"Random Forest (Test Set, F2-Optimal Cutoff = {oof_opt_rf:.2f})",
        output_path=figures_dir / "confusion_matrix_rf_test_optimal.png",
    )
    plot_confusion_matrix(
        test_metrics_lr_default["confusion_matrix"],
        title=f"Logistic Regression (Test Set, Default Cutoff = 0.50)",
        output_path=figures_dir / "confusion_matrix_lr_test_default.png",
    )

    # 2. Precision-Recall Curves
    plot_precision_recall_curves(
        curves_data={
            "Random Forest (Held-Out Test)": (y_test, test_probs_rf, test_metrics_rf_default["average_precision_pr_auc"]),
            "Random Forest (Train OOF)": (y_train, oof_probs_rf, oof_metrics_rf_default["average_precision_pr_auc"]),
            "Logistic Regression (Held-Out Test)": (y_test, test_probs_lr, test_metrics_lr_default["average_precision_pr_auc"]),
            "Logistic Regression (Train OOF)": (y_train, oof_probs_lr, oof_metrics_lr_default["average_precision_pr_auc"]),
        },
        baseline_rate=baseline_failure_rate,
        output_path=figures_dir / "precision_recall_curves.png",
    )

    # 3. ROC Curves
    plot_roc_curves(
        curves_data={
            "Random Forest (Held-Out Test)": (y_test, test_probs_rf, test_metrics_rf_default["roc_auc"]),
            "Random Forest (Train OOF)": (y_train, oof_probs_rf, oof_metrics_rf_default["roc_auc"]),
            "Logistic Regression (Held-Out Test)": (y_test, test_probs_lr, test_metrics_lr_default["roc_auc"]),
            "Logistic Regression (Train OOF)": (y_train, oof_probs_lr, oof_metrics_lr_default["roc_auc"]),
        },
        output_path=figures_dir / "roc_curves.png",
    )

    # 4. Model Comparison Bar Chart
    comparison_summary = [
        {"evaluation_label": "LR (OOF, 0.50)", **oof_metrics_lr_default},
        {"evaluation_label": "LR (Test, 0.50)", **test_metrics_lr_default},
        {"evaluation_label": "RF (OOF, 0.50)", **oof_metrics_rf_default},
        {"evaluation_label": "RF (Test, 0.50)", **test_metrics_rf_default},
        {"evaluation_label": f"RF (Test, {oof_opt_rf:.2f})", **test_metrics_rf_opt},
    ]
    plot_model_comparison_bar(comparison_summary, output_path=figures_dir / "model_comparison.png")

    # Metadata and Complete Report
    metadata = {
        "selected_model": selected_model_name,
        "selected_pipeline_path": (
            "models/random_forest_pipeline.joblib"
            if selected_model_name == "Random Forest"
            else "models/logistic_regression_pipeline.joblib"
        ),
        "selected_threshold": selected_threshold,
        "threshold_selection_method": "F2-score maximization on 5-fold out-of-fold training probabilities",
        "random_state": RANDOM_STATE,
        "features": ALL_PREDICTOR_COLUMNS,
        "transformed_feature_names": feature_names,
        "training_sample_count": len(X_train),
        "test_sample_count": len(X_test),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "safety_disclaimer": SAFETY_DISCLAIMER,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    full_report = {
        "metadata": metadata,
        "dataset_split": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "train_failures": int(y_train.sum()),
            "test_failures": int(y_test.sum()),
            "train_failure_rate_pct": round(baseline_failure_rate * 100, 3),
            "test_failure_rate_pct": round(float(y_test.mean()) * 100, 3),
        },
        "out_of_fold_validation_results": {
            "logistic_regression": {
                "default_threshold_0_5": oof_metrics_lr_default,
                "f2_optimal_threshold": thresh_opt_lr,
            },
            "random_forest": {
                "default_threshold_0_5": oof_metrics_rf_default,
                "f2_optimal_threshold": thresh_opt_rf,
            },
        },
        "held_out_test_results": {
            "logistic_regression": {
                "default_threshold_0_5": test_metrics_lr_default,
                "training_selected_threshold": test_metrics_lr_opt,
            },
            "random_forest": {
                "default_threshold_0_5": test_metrics_rf_default,
                "training_selected_threshold": test_metrics_rf_opt,
            },
        },
        "feature_importance_rf": rf_importances_sorted,
        "feature_coefficients_lr": lr_coefs_sorted,
    }

    metrics_output_path = OUTPUTS_DIR / "model_evaluation_metrics.json"
    with open(metrics_output_path, "w") as f:
        json.dump(full_report, f, indent=2)

    elapsed_time = time.time() - start_time
    print(f"\n[DONE] Model training and evaluation completed successfully in {elapsed_time:.2f}s.")
    return full_report


if __name__ == "__main__":
    train_and_evaluate_models()
