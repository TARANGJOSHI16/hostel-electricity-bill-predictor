"""
generate_data.py
----------------
Generates a realistic, reproducible synthetic dataset for hostel room electricity consumption.

Energy Calculation Formula:
    Energy (kWh) = Power (Watts) * Usage (Hours) / 1000

Appliance Wattage Assumptions (Standard Hostel Fixtures):
    - Ceiling Fan: 75 Watts
    - LED Tube / Light: 25 Watts
    - Air Conditioner (1 - 1.5 Ton, ~80% duty cycle): 1200 Watts effective load
    - Laptop / Laptop Charger: 65 Watts
    - Other Appliances: Direct estimate (kettle, phone charging, iron, trimmer)
"""

import os
import numpy as np
import pandas as pd

# Set fixed random seed for 100% reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Constants for Appliance Wattages (in Watts)
FAN_WATTAGE = 75        # Standard 1200mm ceiling fan
LIGHT_WATTAGE = 25      # LED tube light or warm LED bulb
AC_EFFECTIVE_WATTAGE = 1200  # 1.5 Ton AC with active compressor cycling
LAPTOP_WATTAGE = 65     # Standard student laptop charger


def generate_hostel_data(num_samples: int = 1500) -> pd.DataFrame:
    """
    Generate synthetic room-day records with realistic correlations and noise.

    Parameters:
        num_samples (int): Total number of room-day records to generate (default: 1500).

    Returns:
        pd.DataFrame: DataFrame containing appliance usage features and target daily_kwh.
    """
    # 1. Room Occupants: Typical hostel rooms accommodate 1 to 4 students (occasionally 5-6)
    occupants = np.random.choice([1, 2, 3, 4, 5], size=num_samples, p=[0.20, 0.45, 0.20, 0.10, 0.05])

    # 2. Ceiling Fans: Usually 1 or 2 fans per room (scaled slightly with occupants)
    # A single room typically has 1 fan; 2-4 seater rooms have 1-2 fans.
    fans = np.where(occupants <= 2, np.random.choice([1, 2], p=[0.75, 0.25], size=num_samples),
                    np.random.choice([1, 2, 3], p=[0.20, 0.70, 0.10], size=num_samples))

    # Fan hours: Between 4 and 24 hours/day (high in summer hostels)
    fan_hours = np.random.uniform(6.0, 22.0, size=num_samples)
    fan_hours = np.round(fan_hours, 1)

    # 3. Lights: 1 to 4 lights per room
    lights = np.where(occupants <= 2, np.random.choice([1, 2], p=[0.60, 0.40], size=num_samples),
                      np.random.choice([2, 3, 4], p=[0.50, 0.40, 0.10], size=num_samples))

    # Light hours: 2 to 14 hours/day (active study hours, evenings, night)
    light_hours = np.random.uniform(3.0, 12.0, size=num_samples)
    light_hours = np.round(light_hours, 1)

    # 4. Air Conditioners: Some hostel rooms have AC (0 or 1 unit, rare 2 units)
    # Around 40% of simulated rooms have AC
    has_ac = np.random.choice([0, 1], size=num_samples, p=[0.60, 0.40])
    ac_units = has_ac.copy()

    # AC hours: 0 if no AC; otherwise 3 to 14 hours/day
    ac_hours = np.where(
        ac_units > 0,
        np.round(np.random.uniform(3.0, 12.0, size=num_samples), 1),
        0.0
    )

    # 5. Laptop Hours: Total cumulative laptop operating hours across all occupants
    # Each student typically uses a laptop 2 to 7 hours/day
    avg_hours_per_student = np.random.uniform(2.0, 6.5, size=num_samples)
    laptop_hours = np.round(occupants * avg_hours_per_student, 1)

    # 6. Other Appliances (kWh): Mobile charging, electric kettle, iron, trimmer, mosquito repellant
    # Base load + per-occupant consumption
    base_other = np.random.uniform(0.1, 0.4, size=num_samples)
    per_occupant_other = occupants * np.random.uniform(0.15, 0.35, size=num_samples)
    other_appliance_kwh = np.round(base_other + per_occupant_other, 2)

    # 7. Energy Calculation Using Physical Formula:
    # Energy (kWh) = Power (Watts) * Hours / 1000
    fan_kwh = (fans * fan_hours * FAN_WATTAGE) / 1000.0
    light_kwh = (lights * light_hours * LIGHT_WATTAGE) / 1000.0
    ac_kwh = (ac_units * ac_hours * AC_EFFECTIVE_WATTAGE) / 1000.0
    laptop_kwh = (laptop_hours * LAPTOP_WATTAGE) / 1000.0

    theoretical_kwh = fan_kwh + light_kwh + ac_kwh + laptop_kwh + other_appliance_kwh

    # 8. Realistic Noise: Metering fluctuations, stand-by power loss (+/- 4%)
    noise = np.random.normal(loc=0.0, scale=0.04 * theoretical_kwh, size=num_samples)
    daily_kwh = np.round(theoretical_kwh + noise, 2)

    # Enforce realistic lower bound (no negative or zero consumption)
    daily_kwh = np.maximum(0.20, daily_kwh)

    # Create DataFrame
    data = pd.DataFrame({
        "occupants": occupants,
        "fans": fans,
        "fan_hours": fan_hours,
        "lights": lights,
        "light_hours": light_hours,
        "ac_units": ac_units,
        "ac_hours": ac_hours,
        "laptop_hours": laptop_hours,
        "other_appliance_kwh": other_appliance_kwh,
        "daily_kwh": daily_kwh
    })

    return data


def save_dataset(df: pd.DataFrame, output_path: str = "data/hostel_electricity.csv") -> str:
    """
    Ensure the destination directory exists and save the DataFrame to CSV.
    """
    directory = os.path.dirname(output_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("=" * 60)
    print("Hostel Electricity Predictor - Synthetic Data Generator")
    print("=" * 60)

    dataset_path = "data/hostel_electricity.csv"
    df = generate_hostel_data(num_samples=1500)

    # Save to disk
    save_dataset(df, dataset_path)

    # Print summary statistics
    print(f"\nGenerated {len(df)} records with {df.shape[1]} columns.")
    print("\nFirst 5 Records:")
    print(df.head())

    print("\nSummary Statistics:")
    print(df.describe().T[["mean", "std", "min", "50%", "max"]])
    print("\nDataset generation completed successfully!")
