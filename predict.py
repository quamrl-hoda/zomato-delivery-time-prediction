"""
predict.py  —  Food Delivery Time Predictor
--------------------------------------------
Standalone prediction module.

Pipeline (matches training exactly):
  1. preprocessor.transform(X)               — ColumnTransformer
  2. stacking_model.predict(X_trans)         — output in power-transformed space
  3. power_transformer.inverse_transform()   — back to original minutes scale
"""

import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")

# ── Feature schema (mirrors training data_preprocessing.py exactly) ────────────
NUM_COLS         = ["age", "ratings", "pickup_time_minutes", "distance"]
NOMINAL_CAT_COLS = [
    "weather", "type_of_order", "type_of_vehicle",
    "festival", "city_type", "is_weekend", "order_time_of_day"
]
ORDINAL_CAT_COLS = ["traffic", "distance_type"]
PASSTHROUGH_COLS = ["vehicle_condition", "multiple_deliveries"]   # remainder="passthrough"
ALL_FEATURES     = NUM_COLS + NOMINAL_CAT_COLS + ORDINAL_CAT_COLS + PASSTHROUGH_COLS
TARGET           = "time_taken"

VALID_VALUES = {
    "traffic":       ["low", "medium", "high", "jam"],
    "distance_type": ["short", "medium", "long", "very_long"],
}

# ── Load artifacts ─────────────────────────────────────────────────────────────
preprocessor      = joblib.load(MODELS_DIR / "preprocessor.joblib")
power_transformer = joblib.load(MODELS_DIR / "power_transformer.joblib")
stacking_model    = joblib.load(MODELS_DIR / "stacking_regressor.joblib")


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_input(data: dict) -> list:
    errors = []
    missing = [f for f in ALL_FEATURES if f not in data]
    if missing:
        errors.append(f"Missing fields: {missing}")
    for col, allowed in VALID_VALUES.items():
        if col in data and data[col] not in allowed:
            errors.append(f"'{col}' must be one of {allowed}, got '{data[col]}'")
    return errors


# ── Core pipeline ──────────────────────────────────────────────────────────────
def _run_pipeline(df: pd.DataFrame) -> list:
    # Step 1: preprocess features
    X_trans = preprocessor.transform(df)
    # Step 2: predict in power-transformed target space
    y_pred_transformed = stacking_model.predict(X_trans)
    # Step 3: inverse_transform back to original minutes scale
    y_pred = power_transformer.inverse_transform(
        y_pred_transformed.reshape(-1, 1)
    ).flatten()
    return y_pred.tolist()


# ── Public API ─────────────────────────────────────────────────────────────────
def predict(input_data: dict) -> float:
    """Single-record prediction. Returns predicted time_taken in minutes."""
    errors = validate_input(input_data)
    if errors:
        raise ValueError(f"Validation failed: {errors}")
    df = pd.DataFrame([input_data])[ALL_FEATURES]
    return _run_pipeline(df)[0]


def predict_batch(records: list) -> list:
    """Batch prediction. Returns list of floats (minutes)."""
    for i, rec in enumerate(records):
        errors = validate_input(rec)
        if errors:
            raise ValueError(f"Record {i}: {errors}")
    df = pd.DataFrame(records)[ALL_FEATURES]
    return _run_pipeline(df)


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "age":                 29,
        "ratings":             4.5,
        "pickup_time_minutes": 10,
        "distance":            5.2,
        "weather":             "Sunny",
        "type_of_order":       "Snack",
        "type_of_vehicle":     "motorcycle",
        "festival":            "No",
        "city_type":           "Urban",
        "is_weekend":          "Yes",
        "order_time_of_day":   "Evening",
        "traffic":             "medium",
        "distance_type":       "short",
        "vehicle_condition":   2,
        "multiple_deliveries": 0,
    }
    result = predict(sample)
    print(f"Predicted {TARGET}: {result:.2f} minutes")