"""
Unit tests for the Simulated Sensor Stream Engine.
Verifies determinism, state transitions, bounds checking, scenario dynamics, and leakage prevention.
"""

import numpy as np
import pytest

from src.config import ALL_PREDICTOR_COLUMNS, ID_COLUMNS, LEAKAGE_FAILURE_COLUMNS, TARGET_COLUMN
from src.simulator.stream import (
    SENSOR_BOUNDS,
    VALID_MACHINE_TYPES,
    VALID_SCENARIOS,
    generate_reading_batch,
    init_simulator_state,
    step_simulation,
)


def test_init_simulator_state_defaults():
    """Ensures simulator initializes with expected schema and zero tick."""
    state = init_simulator_state(scenario="normal", machine_type="L", seed=42)

    assert state["tick"] == 0
    assert state["scenario"] == "normal"
    assert state["machine_type"] == "L"
    assert state["seed"] == 42
    assert state["tool_cycle"] == 1
    assert state["tool_replaced"] is False
    assert 295.0 <= state["air_temp"] <= 305.0


def test_init_simulator_invalid_inputs():
    """Ensures invalid scenarios or types raise descriptive ValueErrors."""
    with pytest.raises(ValueError, match="Invalid scenario"):
        init_simulator_state(scenario="catastrophic_meltdown")

    with pytest.raises(ValueError, match="Invalid machine_type"):
        init_simulator_state(machine_type="X")


def test_step_simulation_determinism():
    """Verifies that identical seed and state produce exact identical readings."""
    state1 = init_simulator_state(scenario="normal", machine_type="L", seed=100)
    state2 = init_simulator_state(scenario="normal", machine_type="L", seed=100)

    for _ in range(5):
        state1, out1 = step_simulation(state1)
        state2, out2 = step_simulation(state2)

        assert out1["features"] == out2["features"]
        assert out1["metadata"] == out2["metadata"]
        assert state1 == state2


def test_step_simulation_schema_and_leakage_exclusion():
    """Verifies generated features match ALL_PREDICTOR_COLUMNS with zero leakage."""
    state = init_simulator_state()
    state, output = step_simulation(state)
    features = output["features"]

    # Must match predictor list exactly
    assert list(features.keys()) == ALL_PREDICTOR_COLUMNS

    # Forbidden columns must never be present
    for forbidden in ID_COLUMNS + LEAKAGE_FAILURE_COLUMNS + [TARGET_COLUMN]:
        assert forbidden not in features


def test_sensor_values_strictly_within_empirical_bounds():
    """Verifies that across 50 steps, telemetry stays within empirical bounds."""
    state = init_simulator_state(scenario="elevated_load", machine_type="L", seed=7)

    for _ in range(50):
        state, output = step_simulation(state)
        f = output["features"]

        for col, (b_min, b_max) in SENSOR_BOUNDS.items():
            val = f[col]
            assert b_min <= val <= b_max, f"Column '{col}' value {val} out of bounds [{b_min}, {b_max}]"


def test_scenario_dynamics_differences():
    """Verifies scenarios shift operating centers as expected."""
    df_normal, _ = generate_reading_batch(scenario="normal", n_steps=20, seed=42)
    df_load, _ = generate_reading_batch(scenario="elevated_load", n_steps=20, seed=42)
    df_thermal, _ = generate_reading_batch(scenario="thermal_stress", n_steps=20, seed=42)

    # Elevated load scenario must have higher torque and lower rotational speed
    assert df_load["Torque [Nm]"].median() > df_normal["Torque [Nm]"].median()
    assert df_load["Rotational speed [rpm]"].median() < df_normal["Rotational speed [rpm]"].median()

    # Thermal stress scenario must have higher air temperature
    assert df_thermal["Air temperature [K]"].median() > df_normal["Air temperature [K]"].median()


def test_tool_wear_lifecycle_and_reset():
    """Verifies that tool wear accumulates and resets cleanly when reaching threshold."""
    # Initialize near threshold
    state = init_simulator_state(scenario="high_tool_wear", machine_type="L", seed=42)
    state["tool_wear"] = 244.5  # Just below 245.0 limit

    # Step should trigger reset
    next_state, output = step_simulation(state)

    assert next_state["tool_replaced"] is True
    assert next_state["tool_cycle"] == 2
    assert next_state["tool_wear"] == 0.0
    assert output["metadata"]["tool_replaced"] is True
