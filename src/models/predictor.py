"""
Production prediction interface for Predictive Maintenance Intelligence System.
Safely loads the trained Pipeline artifact and metadata, validates incoming telemetry,
and computes model-estimated anomaly scores and maintenance decision statuses.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_PREDICTOR_COLUMNS,
    FEATURE_RANGES,
    METADATA_PATH,
    SAFETY_DISCLAIMER,
)


class ProductionPredictor:
    """
    Encapsulates the trained scikit-learn Pipeline and its training-selected decision threshold.
    """

    def __init__(
        self,
        metadata_path: Optional[Path] = None,
        pipeline_path: Optional[Path] = None,
    ) -> None:
        self.metadata_path = Path(metadata_path) if metadata_path else METADATA_PATH
        self.pipeline_path = Path(pipeline_path) if pipeline_path else None
        self.model_name: str = ""
        self.threshold: float = 0.5
        self.pipeline: Optional[Pipeline] = None
        self.metadata: Dict[str, Any] = {}

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads and verifies metadata and Joblib pipeline artifact."""
        if not self.metadata_path.exists():
            raise RuntimeError(
                f"Production metadata file not found at '{self.metadata_path}'. "
                "Ensure model training (Phase 5) was executed successfully."
            )

        try:
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to read metadata JSON from '{self.metadata_path}': {e}")

        # Strict validation of metadata content (never silently fall back)
        required_keys = ["selected_model", "selected_pipeline_path", "selected_threshold"]
        for key in required_keys:
            if key not in self.metadata:
                raise RuntimeError(f"Corrupted metadata: missing required key '{key}' in {self.metadata_path}")

        self.model_name = self.metadata["selected_model"]
        try:
            self.threshold = float(self.metadata["selected_threshold"])
        except (ValueError, TypeError):
            raise RuntimeError(f"Invalid threshold value '{self.metadata['selected_threshold']}' in metadata.")

        if self.pipeline_path:
            target_pipe_path = self.pipeline_path
        else:
            raw_path = Path(self.metadata["selected_pipeline_path"])
            if raw_path.exists():
                target_pipe_path = raw_path
            else:
                # Portable resolution: search next to metadata file or in models directory
                candidate = self.metadata_path.parent / raw_path.name
                if candidate.exists():
                    target_pipe_path = candidate
                else:
                    from src.config import MODELS_DIR
                    candidate2 = MODELS_DIR / raw_path.name
                    if candidate2.exists():
                        target_pipe_path = candidate2
                    else:
                        target_pipe_path = raw_path

        if not target_pipe_path.exists():
            raise RuntimeError(
                f"Model pipeline artifact not found at '{target_pipe_path}'. "
                "Ensure models are serialized in models/ directory."
            )

        try:
            self.pipeline = joblib.load(target_pipe_path)
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize Joblib pipeline from '{target_pipe_path}': {e}")

        if not hasattr(self.pipeline, "predict_proba"):
            raise TypeError("Loaded pipeline does not implement 'predict_proba'.")

    def validate_inputs(self, df: pd.DataFrame) -> None:
        """
        Validates schema, completeness, and physical range plausibility of input telemetry.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None.")

        missing_cols = [c for c in ALL_PREDICTOR_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Input data missing required predictor columns: {missing_cols}")

        # Check for non-finite / null values
        null_counts = df[ALL_PREDICTOR_COLUMNS].isnull().sum()
        if null_counts.sum() > 0:
            bad_cols = null_counts[null_counts > 0].to_dict()
            raise ValueError(f"Input data contains null/NaN values in columns: {bad_cols}")

        # Type validity
        valid_types = set(FEATURE_RANGES["Type"]["options"])
        type_vals = set(df["Type"].dropna().unique())
        if not type_vals.issubset(valid_types):
            raise ValueError(f"Invalid 'Type' categories detected: {type_vals - valid_types}. Must be in {valid_types}.")

        # Domain physical checks
        if (df["Air temperature [K]"] <= 0).any():
            raise ValueError("Air temperature contains non-positive values.")
        if (df["Process temperature [K]"] <= 0).any():
            raise ValueError("Process temperature contains non-positive values.")
        if (df["Rotational speed [rpm]"] <= 0).any():
            raise ValueError("Rotational speed contains non-positive values.")
        if (df["Torque [Nm]"] <= 0).any():
            raise ValueError("Torque contains non-positive values.")
        if (df["Tool wear [min]"] < 0).any():
            raise ValueError("Tool wear contains negative values.")

    def predict(
        self,
        input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Computes model-estimated anomaly scores and maintenance decision statuses.

        Parameters
        ----------
        input_data : DataFrame, dict, or list of dicts
            Predictor telemetry.

        Returns
        -------
        list of dict
            Standardized prediction results.
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(f"Unsupported input type '{type(input_data).__name__}'. Use DataFrame or dict.")

        # Validate inputs
        self.validate_inputs(df)

        # Retain only permitted predictor columns in stable order
        X = df[ALL_PREDICTOR_COLUMNS].copy()

        # Generate estimated scores (probabilities for class 1)
        scores = self.pipeline.predict_proba(X)[:, 1]
        labels = (scores >= self.threshold).astype(int)

        results: List[Dict[str, Any]] = []
        for idx in range(len(df)):
            score = float(scores[idx])
            label = int(labels[idx])

            if label == 1:
                status = "MAINTENANCE_REQUIRED"
                risk_level = "HIGH"
            elif score >= (self.threshold * 0.70):
                status = "ATTENTION_ADVISED"
                risk_level = "ELEVATED"
            else:
                status = "NORMAL_OPERATION"
                risk_level = "LOW"

            results.append(
                {
                    "model_name": self.model_name,
                    "decision_threshold": self.threshold,
                    "estimated_anomaly_score": round(score, 4),
                    "predicted_label": label,
                    "status": status,
                    "risk_level": risk_level,
                    "score_interpretation": (
                        "Model-estimated score derived from the synthetic UCI AI4I 2020 benchmark. "
                        "This is an educational statistical estimate, not a real-world calibrated failure probability."
                    ),
                    "safety_disclaimer": SAFETY_DISCLAIMER,
                }
            )

        return results


# Module-level cached predictor instance
_PREDICTOR_INSTANCE: Optional[ProductionPredictor] = None


def get_predictor() -> ProductionPredictor:
    """Provides a thread-safe singleton-like instance of ProductionPredictor."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = ProductionPredictor()
    return _PREDICTOR_INSTANCE


def predict_reading(
    reading: Dict[str, Any],
) -> Dict[str, Any]:
    """Convenience function to predict on a single sensor reading dictionary."""
    predictor = get_predictor()
    results = predictor.predict(reading)
    return results[0]


if __name__ == "__main__":
    from src.simulator.stream import init_simulator_state, step_simulation

    print("\n" + "=" * 80)
    print(" PREDICTIVE MAINTENANCE: SENSOR SIMULATOR & PREDICTOR SMOKE TEST")
    print(" [Notice: All telemetry is synthetic simulation, not real machine hardware]")
    print("=" * 80)

    predictor = ProductionPredictor()
    print(f"Loaded Model: {predictor.model_name} (Decision Threshold = {predictor.threshold:.2f})")
    print(f"Features Evaluated: {ALL_PREDICTOR_COLUMNS}\n")

    scenarios = ["normal", "elevated_load", "thermal_stress", "high_tool_wear"]

    for sc in scenarios:
        print(f"--- Simulating Scenario: {sc.upper()} (Machine Type: L) ---")
        state = init_simulator_state(scenario=sc, machine_type="L", seed=42)

        for step in range(3):
            state, output = step_simulation(state)
            features = output["features"]
            meta = output["metadata"]

            pred = predictor.predict(features)[0]

            print(
                f" Tick {meta['tick']:02d} | "
                f"Air: {features['Air temperature [K]']}K, "
                f"Proc: {features['Process temperature [K]']}K, "
                f"RPM: {features['Rotational speed [rpm]']}, "
                f"Torque: {features['Torque [Nm]']}Nm, "
                f"Wear: {features['Tool wear [min]']}min | "
                f"Score: {pred['estimated_anomaly_score']:.4f} -> [{pred['status']}]"
            )
        print()

    print("=" * 80)
    print("Smoke test successfully completed without hardware connectivity or data fabrication.")
    print("=" * 80 + "\n")
