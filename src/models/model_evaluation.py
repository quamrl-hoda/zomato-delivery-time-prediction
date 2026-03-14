import pandas as pd
import joblib
import logging
import mlflow
import mlflow.data
import mlflow.sklearn
import dagshub
import shutil
import json
import os
import time
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv

# ── Environment & tracking setup ────────────────────────────────────────────

load_dotenv()

# FIX 1: Strip whitespace/newlines from token to avoid header-injection errors
dagshub_token = os.getenv("DAGSHUB_TOKEN", "").strip()
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN is not set in your .env file")
os.environ["DAGSHUB_USER_TOKEN"] = dagshub_token

# FIX 2: Increase upload timeout and retries before any MLflow calls
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "300"
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "5"
os.environ["MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"] = "600"

# FIX 3: dagshub.init() already sets the tracking URI internally —
#         remove the duplicate mlflow.set_tracking_uri() that pointed to the
#         wrong repo name ("...-system.mlflow" vs the actual repo name).
dagshub.init(
    repo_owner="quamrl-hoda",
    repo_name="zomato-delivery-time-prediction",
    mlflow=True
)

mlflow.set_experiment("zomato-delivery-time-prediction")

TARGET = "time_taken"

# ── Logger ───────────────────────────────────────────────────────────────────

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(handler)

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_data(data_path: Path) -> pd.DataFrame:
    # FIX 4: raise the error instead of silently returning an unbound variable
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        raise


def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def load_model(model_path: Path):
    return joblib.load(model_path)


def save_model_info(save_json_path, run_id, artifact_path, model_name):
    info_dict = {
        "run_id": run_id,
        "artifact_path": artifact_path,
        "model_name": model_name,
    }
    with open(save_json_path, "w") as f:
        json.dump(info_dict, f, indent=4)


def log_artifacts_with_retry(local_dir: Path, artifact_path: str,
                              retries: int = 3, wait: int = 15):
    """Upload artifact directory to MLflow with retry on connection reset."""
    for attempt in range(1, retries + 1):
        try:
            mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)
            logger.info("Model artifacts uploaded successfully.")
            return
        except Exception as e:
            logger.warning(f"Upload attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)
    raise RuntimeError("All artifact upload attempts failed.")


def log_single_artifact_with_retry(path: Path, retries: int = 3, wait: int = 10):
    """Upload a single artifact file with retry."""
    for attempt in range(1, retries + 1):
        try:
            mlflow.log_artifact(str(path))
            return
        except Exception as e:
            logger.warning(f"Artifact upload attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(wait)
    logger.error(f"Failed to upload artifact: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root_path = Path(__file__).parent.parent.parent

    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    test_data_path  = root_path / "data" / "processed" / "test_trans.csv"
    model_path      = root_path / "models" / "model.joblib"

    # Load data
    train_data = load_data(train_data_path)
    logger.info("Train data loaded successfully")
    test_data = load_data(test_data_path)
    logger.info("Test data loaded successfully")

    X_train, y_train = make_X_and_y(train_data, TARGET)
    X_test,  y_test  = make_X_and_y(test_data,  TARGET)
    logger.info("Data split completed")

    # FIX 5: Cast integer columns to float64 to silence MLflow schema warnings
    int_cols = X_train.select_dtypes(include="int64").columns.tolist()
    if int_cols:
        X_train[int_cols] = X_train[int_cols].astype("float64")
        X_test[int_cols]  = X_test[int_cols].astype("float64")
        logger.info(f"Cast {len(int_cols)} integer column(s) to float64: {int_cols}")

    # Load model
    model = load_model(model_path)
    logger.info("Model loaded successfully")

    # Predictions & metrics
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae  = mean_absolute_error(y_test,  y_test_pred)
    train_r2  = r2_score(y_train, y_train_pred)
    test_r2   = r2_score(y_test,  y_test_pred)
    logger.info(f"train_mae={train_mae:.4f}  test_mae={test_mae:.4f}  "
                f"train_r2={train_r2:.4f}  test_r2={test_r2:.4f}")

    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    mean_cv_score = -cv_scores.mean()
    logger.info(f"Cross-validation complete — mean CV MAE: {mean_cv_score:.4f}")

    # ── MLflow run ───────────────────────────────────────────────────────────
    with mlflow.start_run() as run:

        mlflow.set_tag("model", "Food Delivery Time Regressor")
        mlflow.log_params(model.get_params())

        mlflow.log_metrics({
            "train_mae":      train_mae,
            "test_mae":       test_mae,
            "train_r2":       train_r2,
            "test_r2":        test_r2,
            "mean_cv_score":  mean_cv_score,
        })
        mlflow.log_metrics({f"CV_{i}": -s for i, s in enumerate(cv_scores)})

        # Dataset logging
        train_input = mlflow.data.from_pandas(train_data, targets=TARGET)
        test_input  = mlflow.data.from_pandas(test_data,  targets=TARGET)
        mlflow.log_input(dataset=train_input, context="training")
        mlflow.log_input(dataset=test_input,  context="validation")

        # Model signature
        sample      = X_train.sample(20, random_state=42)
        signature   = mlflow.models.infer_signature(
            model_input=sample,
            model_output=model.predict(sample)
        )

        # FIX 6: Save model locally then upload with retry
        logger.info("Starting model artifact upload to MLflow...")
        model_save_path = root_path / "models" / "mlflow_model"
        try:
            if model_save_path.exists():
                shutil.rmtree(model_save_path)

            mlflow.sklearn.save_model(
                sk_model=model,
                path=str(model_save_path),
                signature=signature
            )

            log_artifacts_with_retry(model_save_path, artifact_path="model")

        finally:
            # Always clean up the temp folder
            if model_save_path.exists():
                shutil.rmtree(model_save_path)

        # FIX 7: Check each extra artifact file exists before logging
        extra_artifacts = [
            root_path / "models" / "stacking_regressor.joblib",
            root_path / "models" / "power_transformer.joblib",
            root_path / "models" / "preprocessor.joblib",
        ]
        for artifact_path in extra_artifacts:
            if artifact_path.exists():
                log_single_artifact_with_retry(artifact_path)
            else:
                logger.warning(f"Artifact not found, skipping: {artifact_path}")

        # Verify
        client    = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run.info.run_id)
        logger.info(f"Artifacts in run: {[a.path for a in artifacts]}")

        artifact_uri = mlflow.get_artifact_uri()
        logger.info("MLflow logging complete")

    # Save run info
    run_id        = run.info.run_id
    save_json_path = root_path / "run_information.json"
    save_model_info(
        save_json_path=save_json_path,
        run_id=run_id,
        artifact_path=artifact_uri,
        model_name="model"
    )
    logger.info(f"Model information saved → run_id: {run_id}")