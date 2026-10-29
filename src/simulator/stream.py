"""
Simulated sensor stream engine for Predictive Maintenance Intelligence System.
Generates physically plausible, changing sensor telemetry within empirical dataset bounds.
Completely software-only, deterministic, serializable, and independent of frontend/hardware.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import ALL_PREDICTOR_COLUMNS, FEATURE_RANGES

# Available simulation operational scenarios
VALID_SCENARIOS = ["normal", "elevated_load", "thermal_stress", "high_tool_wear"]
VALID_MACHINE_TYPES = ["L", "M", "H"]

# Empirical dataset operating bounds (from actual AI4I 2020 dataset)
SENSOR_BOUNDS = {
    "Air temperature [K]": (295.3, 304.5),
    "Process temperature [K]": (305.7, 313.8),
    "Rotational speed [rpm]": (1168.0, 2886.0),
    "Torque [Nm]": (3.8, 76.6),
    "Tool wear [min]": (0.0, 253.0),
}


def init_simulator_state(
    scenario: str = "normal",
    machine_type: str = "L",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Initializes a serializable simulator state dictionary.

    Parameters
    ----------
    scenario : str
        Operational scenario ('normal', 'elevated_load', 'thermal_stress', 'high_tool_wear').
    machine_type : str
        Quality variant ('L', 'M', or 'H').
    seed : int
        Base seed for deterministic random-walk transitions.

    Returns
    -------
    dict
        State dictionary containing all parameters needed to step the simulation.
    """
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"Invalid scenario '{scenario}'. Must be one of {VALID_SCENARIOS}.")

    if machine_type not in VALID_MACHINE_TYPES:
        raise ValueError(f"Invalid machine_type '{machine_type}'. Must be one of {VALID_MACHINE_TYPES}.")

    # Scenario-specific base operating centers
    if scenario == "normal":
        air_temp = 300.0
        proc_temp = 310.0
        speed = 1500.0
        torque = 39.5
        tool_wear = 10.0
    elif scenario == "elevated_load":
        air_temp = 300.5
        proc_temp = 310.8
        speed = 1320.0
        torque = 58.0
        tool_wear = 45.0
    elif scenario == "thermal_stress":
        air_temp = 303.5
        proc_temp = 312.3
        speed = 1460.0
        torque = 46.0
        tool_wear = 70.0
    elif scenario == "high_tool_wear":
        air_temp = 301.0
        proc_temp = 310.9
        speed = 1440.0
        torque = 48.0
        tool_wear = 210.0

    return {
        "tick": 0,
        "seed": int(seed),
        "scenario": scenario,
        "machine_type": machine_type,
        "air_temp": float(air_temp),
        "process_temp": float(proc_temp),
        "rotational_speed": float(speed),
        "torque": float(torque),
        "tool_wear": float(tool_wear),
        "tool_cycle": 1,
        "tool_replaced": False,
    }


def step_simulation(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Advances the simulation by one tick using mean-reverting stochastic walk.

    Parameters
    ----------
    state : dict
        Current simulator state.

    Returns
    -------
    next_state : dict
        Updated simulator state.
    reading : dict
        Standardized sensor reading with exact keys matching ALL_PREDICTOR_COLUMNS,
        plus simulation metadata.
    """
    tick = state["tick"]
    seed = state["seed"]
    scenario = state["scenario"]
    machine_type = state["machine_type"]

    # Seed-per-tick deterministic PRNG (reproducible anywhere)
    rng = np.random.default_rng(seed + tick * 101)

    # Operating targets based on scenario
    if scenario == "normal":
        target_air = 300.0
        target_delta_t = 10.0
        target_torque = 39.8
        target_speed = 1505.0
        wear_rate = 1.8 if machine_type == "H" else (2.4 if machine_type == "M" else 3.0)
    elif scenario == "elevated_load":
        target_air = 300.8
        target_delta_t = 10.3
        target_torque = 59.5
        target_speed = 1315.0
        wear_rate = 2.5 if machine_type == "H" else (3.2 if machine_type == "M" else 4.0)
    elif scenario == "thermal_stress":
        target_air = 303.6
        target_delta_t = 8.8  # Deficient heat dissipation
        target_torque = 47.0
        target_speed = 1450.0
        wear_rate = 2.0 if machine_type == "H" else (2.6 if machine_type == "M" else 3.2)
    elif scenario == "high_tool_wear":
        target_air = 301.2
        target_delta_t = 9.8
        target_torque = 51.0
        target_speed = 1420.0
        wear_rate = 2.2 if machine_type == "H" else (2.8 if machine_type == "M" else 3.5)

    # 1. Air temperature random walk with mean-reversion
    reversion_speed = 0.15
    air_drift = reversion_speed * (target_air - state["air_temp"])
    air_noise = rng.normal(0.0, 0.12)
    new_air = state["air_temp"] + air_drift + air_noise

    # 2. Process temperature coupled to air temp
    delta_t_noise = rng.normal(0.0, 0.08)
    new_delta_t = target_delta_t + delta_t_noise
    new_proc = new_air + new_delta_t

    # 3. Torque random walk
    torque_drift = reversion_speed * (target_torque - state["torque"])
    torque_noise = rng.normal(0.0, 0.8)
    new_torque = state["torque"] + torque_drift + torque_noise

    # 4. Rotational speed inversely coupled to torque (P = tau * omega)
    # Power around 2860 W base with slight noise
    speed_drift = reversion_speed * (target_speed - state["rotational_speed"])
    speed_noise = rng.normal(0.0, 12.0)
    new_speed = state["rotational_speed"] + speed_drift + speed_noise

    # 5. Tool wear progression
    wear_inc = rng.uniform(wear_rate * 0.8, wear_rate * 1.2)
    new_wear = state["tool_wear"] + wear_inc

    tool_replaced = False
    tool_cycle = state["tool_cycle"]

    # Tool wear reset when maximum lifecycle is reached (~245 min observed)
    if new_wear >= 245.0:
        new_wear = 0.0
        tool_cycle += 1
        tool_replaced = True

    # Clamp strictly within empirical dataset bounds
    new_air = float(np.clip(new_air, SENSOR_BOUNDS["Air temperature [K]"][0], SENSOR_BOUNDS["Air temperature [K]"][1]))
    new_proc = float(np.clip(new_proc, SENSOR_BOUNDS["Process temperature [K]"][0], SENSOR_BOUNDS["Process temperature [K]"][1]))
    new_speed = float(np.clip(new_speed, SENSOR_BOUNDS["Rotational speed [rpm]"][0], SENSOR_BOUNDS["Rotational speed [rpm]"][1]))
    new_torque = float(np.clip(new_torque, SENSOR_BOUNDS["Torque [Nm]"][0], SENSOR_BOUNDS["Torque [Nm]"][1]))
    new_wear = float(np.clip(new_wear, SENSOR_BOUNDS["Tool wear [min]"][0], SENSOR_BOUNDS["Tool wear [min]"][1]))

    next_state = {
        "tick": tick + 1,
        "seed": seed,
        "scenario": scenario,
        "machine_type": machine_type,
        "air_temp": new_air,
        "process_temp": new_proc,
        "rotational_speed": new_speed,
        "torque": new_torque,
        "tool_wear": new_wear,
        "tool_cycle": tool_cycle,
        "tool_replaced": tool_replaced,
    }

    # Model predictor payload (Strictly 6 permitted columns)
    reading = {
        "Type": machine_type,
        "Air temperature [K]": round(new_air, 2),
        "Process temperature [K]": round(new_proc, 2),
        "Rotational speed [rpm]": int(round(new_speed)),
        "Torque [Nm]": round(new_torque, 2),
        "Tool wear [min]": int(round(new_wear)),
    }

    # Diagnostic metadata
    metadata = {
        "tick": tick + 1,
        "scenario": scenario,
        "tool_cycle": tool_cycle,
        "tool_replaced": tool_replaced,
        "is_simulated": True,
        "disclaimer": "Simulated approximation based on UCI AI4I 2020 distributions; not real machinery hardware.",
    }

    return next_state, {"features": reading, "metadata": metadata}


def generate_reading_batch(
    scenario: str = "normal",
    machine_type: str = "L",
    n_steps: int = 10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Convenience function to generate a batch of sequential simulated readings.

    Returns
    -------
    features_df : pd.DataFrame
        DataFrame with columns exactly matching ALL_PREDICTOR_COLUMNS.
    metadata_list : list of dict
        Simulation metadata for each tick.
    """
    state = init_simulator_state(scenario=scenario, machine_type=machine_type, seed=seed)
    features_list = []
    metadata_list = []

    for _ in range(n_steps):
        state, output = step_simulation(state)
        features_list.append(output["features"])
        metadata_list.append(output["metadata"])

    df = pd.DataFrame(features_list)
    return df, metadata_list
