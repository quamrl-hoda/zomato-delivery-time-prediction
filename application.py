"""
app.py  —  Food Delivery Time Predictor · Flask API
-----------------------------------------------------
Folder structure:
  app.py
  predict.py
  models/
      preprocessor.joblib
      power_transformer.joblib
      stacking_regressor.joblib
  templates/
      index.html
  static/
      css/style.css
      js/main.js

Run:  python app.py
UI:   http://localhost:5000

Pipeline (matches training exactly):
  1. preprocessor.transform(X)         → ColumnTransformer (MinMax + OHE + Ordinal + passthrough)
  2. stacking_model.predict(X_trans)   → raw prediction in power-transformed space
  3. power_transformer.inverse_transform(pred) → actual minutes
"""

import warnings
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.exceptions import InconsistentVersionWarning
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("model_api")

# ── Feature schema (mirrors training data_preprocessing.py exactly) ────────────
# Handled by MinMaxScaler
NUM_COLS         = ["age", "ratings", "pickup_time_minutes", "distance"]
# Handled by OneHotEncoder
NOMINAL_CAT_COLS = [
    "weather", "type_of_order", "type_of_vehicle",
    "festival", "city_type", "is_weekend", "order_time_of_day"
]
# Handled by OrdinalEncoder
ORDINAL_CAT_COLS = ["traffic", "distance_type"]
# remainder="passthrough" — passed through as-is by ColumnTransformer
PASSTHROUGH_COLS = ["vehicle_condition", "multiple_deliveries"]

# Input column order for DataFrame construction
ALL_FEATURES = NUM_COLS + NOMINAL_CAT_COLS + ORDINAL_CAT_COLS + PASSTHROUGH_COLS
TARGET       = "time_taken"

VALID_VALUES = {
    "traffic":       ["low", "medium", "high", "jam"],
    "distance_type": ["short", "medium", "long", "very_long"],
}

# ── Load artifacts ─────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")
logger.info("Loading model artifacts …")
preprocessor      = joblib.load(MODELS_DIR / "preprocessor.joblib")
power_transformer = joblib.load(MODELS_DIR / "power_transformer.joblib")
stacking_model    = joblib.load(MODELS_DIR / "stacking_regressor.joblib")
logger.info("All artifacts loaded ✓")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────
def validate(data: dict) -> list:
    errors = []
    missing = [f for f in ALL_FEATURES if f not in data]
    if missing:
        errors.append(f"Missing fields: {missing}")
    for col, allowed in VALID_VALUES.items():
        if col in data and data[col] not in allowed:
            errors.append(f"'{col}' must be one of {allowed}, got '{data[col]}'")
    return errors


def run_pipeline(df: pd.DataFrame) -> list:
    """
    Correct inference pipeline:
      1. preprocessor.transform   — ColumnTransformer (fitted on X_train)
      2. stacking_model.predict   — raw output in power-transformed target space
      3. power_transformer.inverse_transform — convert back to original minutes scale
    """
    # Step 1: preprocess features
    X_trans = preprocessor.transform(df)

    # Step 2: model predicts in transformed-target space
    y_pred_transformed = stacking_model.predict(X_trans)

    # Step 3: inverse transform back to original scale (minutes)
    y_pred = power_transformer.inverse_transform(
        y_pred_transformed.reshape(-1, 1)
    ).flatten()

    return y_pred.tolist()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Single-record prediction."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Empty request body"}), 400

        errors = validate(data)
        if errors:
            return jsonify({"error": errors}), 422

        df   = pd.DataFrame([data])[ALL_FEATURES]
        pred = run_pipeline(df)[0]
        logger.info(f"Single prediction → {pred:.2f} min")
        return jsonify({
            "prediction": round(pred, 4),
            "target":     TARGET,
            "unit":       "minutes"
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Batch prediction."""
    try:
        body    = request.get_json(force=True)
        records = body.get("records", [])

        if not records:
            return jsonify({"error": "No records provided"}), 400

        for i, rec in enumerate(records):
            errors = validate(rec)
            if errors:
                return jsonify({"error": f"Record {i}: {errors}"}), 422

        df    = pd.DataFrame(records)[ALL_FEATURES]
        preds = [round(p, 4) for p in run_pipeline(df)]
        logger.info(f"Batch prediction → {len(preds)} records")
        return jsonify({
            "predictions": preds,
            "count":       len(preds),
            "target":      TARGET,
            "unit":        "minutes"
        })

    except Exception as e:
        logger.error(f"Batch error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "stacking_regressor", "target": TARGET})


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)