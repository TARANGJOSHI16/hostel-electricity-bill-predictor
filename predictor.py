"""
predictor.py
------------
Prediction engine for the Hostel Electricity Consumption Predictor.

Provides functions to:
1. Load the trained machine learning model from disk.
2. Validate user input features.
3. Predict estimated daily and monthly electricity consumption in kWh (units).
4. Compute an appliance-wise energy breakdown and energy-saving recommendations.

Note:
    1 kWh (Kilowatt-hour) = 1 Electricity Unit (standard metric across electricity boards).
"""

import os
import joblib
import numpy as np
import pandas as pd

# Default model path
DEFAULT_MODEL_PATH = "models/electricity_model.joblib"

# Appliance wattages for analytical breakdown
FAN_WATTAGE = 75
LIGHT_WATTAGE = 25
AC_WATTAGE = 1200
LAPTOP_WATTAGE = 65

FEATURE_ORDER = [
    "occupants",
    "fans",
    "fan_hours",
    "lights",
    "light_hours",
    "ac_units",
    "ac_hours",
    "laptop_hours",
    "other_appliance_kwh"
]


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the saved model package from disk.

    Returns:
        dict: The model payload dictionary containing 'model', 'feature_cols', and 'metrics'.

    Raises:
        FileNotFoundError: If the model file has not yet been generated.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            f"Please run 'python train_model.py' first to train and save the model."
        )

    payload = joblib.load(model_path)
    return payload


def calculate_breakdown(inputs: dict) -> dict:
    """
    Compute theoretical appliance-wise energy consumption in kWh (units) per day.

    Formula: Energy (kWh) = Power (Watts) * Hours / 1000
    """
    fan_kwh = (inputs["fans"] * inputs["fan_hours"] * FAN_WATTAGE) / 1000.0
    light_kwh = (inputs["lights"] * inputs["light_hours"] * LIGHT_WATTAGE) / 1000.0
    ac_kwh = (inputs["ac_units"] * inputs["ac_hours"] * AC_WATTAGE) / 1000.0
    laptop_kwh = (inputs["laptop_hours"] * LAPTOP_WATTAGE) / 1000.0
    other_kwh = float(inputs["other_appliance_kwh"])

    breakdown = {
        "Ceiling Fans": round(fan_kwh, 2),
        "Lights": round(light_kwh, 2),
        "Air Conditioner": round(ac_kwh, 2),
        "Laptops": round(laptop_kwh, 2),
        "Other Appliances": round(other_kwh, 2)
    }

    total_breakdown = sum(breakdown.values())
    breakdown_pct = {}
    for app, kwh in breakdown.items():
        breakdown_pct[app] = round((kwh / total_breakdown * 100), 1) if total_breakdown > 0 else 0.0

    # Determine highest consuming appliance
    highest_appliance = max(breakdown, key=breakdown.get)

    tips = {
        "Air Conditioner": (
            "Air conditioning consumes the largest share of energy in this room. "
            "Set the thermostat to 24°C-26°C instead of 18°C-20°C and use an off-timer at night "
            "to reduce consumption by up to 25%."
        ),
        "Ceiling Fans": (
            "Ceiling fans run for long durations. Turn off fans whenever leaving the room "
            "and ensure fan blades are cleaned regularly for optimal airflow efficiency."
        ),
        "Lights": (
            "Maximize daylight during afternoon study hours and ensure LED fixtures "
            "are turned off when leaving the room."
        ),
        "Laptops": (
            "Unplug laptop chargers when batteries reach 100% and enable energy-saver sleep mode "
            "during breaks to eliminate phantom draw."
        ),
        "Other Appliances": (
            "High wattage appliances like electric kettles or irons contribute significantly. "
            "Avoid leaving electric kettles on keep-warm mode unnecessarily."
        )
    }

    return {
        "breakdown_kwh": breakdown,
        "breakdown_pct": breakdown_pct,
        "highest_appliance": highest_appliance,
        "tip": tips.get(highest_appliance, "Turn off appliances when not actively in use to save energy.")
    }


def predict_electricity(
    input_data: dict,
    days: int = 30,
    model_payload: dict = None
) -> dict:
    """
    Predict daily and monthly electricity consumption for a hostel room.

    Parameters:
        input_data (dict): Dictionary with all required appliance features.
        days (int): Number of days for monthly calculation (default: 30).
        model_payload (dict, optional): Pre-loaded model payload. If None, loaded from disk.

    Returns:
        dict: Formatted predictions, monthly estimates, breakdown, and saving tips.
    """
    if model_payload is None:
        model_payload = load_model()

    # Extract the actual scikit-learn model object
    model = model_payload.get("model", model_payload)

    # Validate inputs
    occupants = max(1, int(input_data.get("occupants", 1)))
    fans = max(0, int(input_data.get("fans", 0)))
    fan_hours = min(24.0, max(0.0, float(input_data.get("fan_hours", 0.0))))
    lights = max(0, int(input_data.get("lights", 0)))
    light_hours = min(24.0, max(0.0, float(input_data.get("light_hours", 0.0))))
    ac_units = max(0, int(input_data.get("ac_units", 0)))
    ac_hours = min(24.0, max(0.0, float(input_data.get("ac_hours", 0.0))))
    laptop_hours = max(0.0, float(input_data.get("laptop_hours", 0.0)))
    other_appliance_kwh = max(0.0, float(input_data.get("other_appliance_kwh", 0.0)))
    days = max(1, int(days))

    sanitized_inputs = {
        "occupants": occupants,
        "fans": fans,
        "fan_hours": fan_hours,
        "lights": lights,
        "light_hours": light_hours,
        "ac_units": ac_units,
        "ac_hours": ac_hours,
        "laptop_hours": laptop_hours,
        "other_appliance_kwh": other_appliance_kwh
    }

    # Format input into single-row DataFrame with exact training columns
    input_df = pd.DataFrame([sanitized_inputs])[FEATURE_ORDER]

    # Predict daily kWh using the ML model
    pred_raw = model.predict(input_df)[0]
    daily_kwh = round(float(np.maximum(0.1, pred_raw)), 2)

    # Compute monthly consumption
    monthly_kwh = round(daily_kwh * days, 2)

    # Calculate analytical appliance breakdown and tips
    breakdown_info = calculate_breakdown(sanitized_inputs)

    return {
        "daily_kwh": daily_kwh,
        "monthly_kwh": monthly_kwh,
        "days": days,
        "sanitized_inputs": sanitized_inputs,
        "breakdown_kwh": breakdown_info["breakdown_kwh"],
        "breakdown_pct": breakdown_info["breakdown_pct"],
        "highest_appliance": breakdown_info["highest_appliance"],
        "energy_saving_tip": breakdown_info["tip"],
        "model_name": model_payload.get("model_name", "Machine Learning Regression")
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Hostel Electricity Predictor - Prediction Test")
    print("=" * 60)

    # Sample hostel room test case: 2 occupants, 2 fans, 1 AC, 2 laptops
    sample_room = {
        "occupants": 2,
        "fans": 2,
        "fan_hours": 12.0,
        "lights": 2,
        "light_hours": 8.0,
        "ac_units": 1,
        "ac_hours": 6.0,
        "laptop_hours": 8.0,
        "other_appliance_kwh": 0.80
    }

    results = predict_electricity(sample_room, days=30)

    print("\nRoom Parameters:")
    for key, val in results["sanitized_inputs"].items():
        print(f"  {key:<20}: {val}")

    print("\nPrediction Results:")
    print(f"  Daily Consumption   : {results['daily_kwh']} kWh (units)")
    print(f"  Monthly Consumption : {results['monthly_kwh']} kWh (units) [over {results['days']} days]")
    print(f"  Highest Consumer    : {results['highest_appliance']}")
    print(f"  Tip                 : {results['energy_saving_tip']}")

    print("\nAppliance Breakdown:")
    for app, kwh in results["breakdown_kwh"].items():
        pct = results["breakdown_pct"][app]
        print(f"  {app:<20}: {kwh:5.2f} kWh ({pct:5.1f}%)")

    print("\nPredictor test passed successfully!")
    print("=" * 60)
