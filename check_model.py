import mlflow
from mlflow import MlflowClient

tracking_uri = "https://dagshub.com/quamrl-hoda/zomato-delivery-time-prediction-system.mlflow"
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri=tracking_uri)

try:
    versions = client.get_latest_versions(name="model", stages=["Staging", "Production"])
    for v in versions:
        print(f"Name: {v.name}, Version: {v.version}, Stage: {v.current_stage}")
except Exception as e:
    print(f"Error: {e}")
