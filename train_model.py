"""
train_model.py
--------------
Trains a machine learning regression model to predict daily hostel electricity consumption.

Steps:
1. Load dataset from data/hostel_electricity.csv (auto-generate if missing).
2. Split data into training (80%) and testing (20%) sets.
3. Train and compare Linear Regression and Random Forest Regressor models.
4. Evaluate with standard metrics: MAE, RMSE, and R2 score.
5. Save the best model and metadata to models/electricity_model.joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from generate_data import generate_hostel_data, save_dataset

DATA_PATH = "data/hostel_electricity.csv"
MODEL_PATH = "models/electricity_model.joblib"
RANDOM_SEED = 42

FEATURE_COLS = [
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
TARGET_COL = "daily_kwh"


def load_or_create_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load data from CSV. If the file does not exist, generate it automatically.
    """
    if not os.path.exists(filepath):
        print(f"Dataset not found at '{filepath}'. Auto-generating now...")
        df = generate_hostel_data(num_samples=1500)
        save_dataset(df, filepath)
    else:
        print(f"Loading existing dataset from '{filepath}'...")
        df = pd.read_csv(filepath)

    return df


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Calculate and print MAE, RMSE, and R2 score for a trained model.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {name} Performance ---")
    print(f"Mean Absolute Error (MAE) : {mae:.4f} kWh (units)")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} kWh (units)")
    print(f"R-squared Score (R2)      : {r2:.4f} ({r2 * 100:.2f}% variance explained)")

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4)
    }


def train_and_save_model():
    """
    Main training workflow: Load data, split, train models, select best, and save.
    """
    print("=" * 60)
    print("Hostel Electricity Predictor - Model Training")
    print("=" * 60)

    # 1. Load data
    df = load_or_create_data(DATA_PATH)

    # 2. Separate features (X) and target (y)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # 3. 80-20 Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED
    )
    print(f"\nDataset split into:")
    print(f"  - Training samples: {len(X_train)} (80%)")
    print(f"  - Testing samples : {len(X_test)} (20%)")

    # 4. Train Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model("Linear Regression", lr_model, X_test, y_test)

    # 5. Train Random Forest Regressor Model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model("Random Forest Regressor", rf_model, X_test, y_test)

    # 6. Select Best Model (based on R2 score)
    if rf_metrics["r2"] >= lr_metrics["r2"]:
        best_model = rf_model
        best_name = "Random Forest Regressor"
        best_metrics = rf_metrics
        feature_importances = dict(zip(FEATURE_COLS, rf_model.feature_importances_))
    else:
        best_model = lr_model
        best_name = "Linear Regression"
        best_metrics = lr_metrics
        feature_importances = dict(zip(FEATURE_COLS, lr_model.coef_))

    print(f"\nSelected Best Model: {best_name}")

    # Display Feature Importance / Coefficients
    print("\nFeature Impact on Daily Electricity Consumption:")
    sorted_features = sorted(feature_importances.items(), key=lambda item: abs(item[1]), reverse=True)
    for feat, imp in sorted_features:
        print(f"  - {feat:<20}: {imp:+.4f}")

    # 7. Ensure models/ directory exists and save model
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"\nCreated directory: {model_dir}")

    # Package model with metadata for clean integration with predictor and app
    payload = {
        "model": best_model,
        "model_name": best_name,
        "feature_cols": FEATURE_COLS,
        "metrics": best_metrics,
        "feature_importances": feature_importances,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    joblib.dump(payload, MODEL_PATH)
    print(f"\nTrained model and metadata successfully saved to: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save_model()
