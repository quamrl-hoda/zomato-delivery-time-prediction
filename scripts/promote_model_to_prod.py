import mlflow
import dagshub
import json
from mlflow import MlflowClient

import os
from dagshub.auth import add_app_token

# Try to get token from multiple common environment variables
token = os.getenv("DAGSHUB_TOKEN") or os.getenv("DAGSHUB_USER_TOKEN")

if token:
    add_app_token(token)
else:
    print("WARNING: DagsHub token not found in environment variables.")

dagshub.init(repo_owner='quamrl-hoda', 
             repo_name='zomato-delivery-time-prediction-system', 
             mlflow=True)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/quamrl-hoda/zomato-delivery-time-prediction-system.mlflow")

def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


# get model name
model_name = load_model_information("run_information.json")["model_name"]
stage = "Staging"

# get the latest version from staging stage
client = MlflowClient()

# get the latest version of model in staging
latest_versions = client.get_latest_versions(name=model_name,stages=[stage])

latest_model_version_staging = latest_versions[0].version

# promotion stage
promotion_stage = "Production"

client.transition_model_version_stage(
    name=model_name,
    version=latest_model_version_staging,
    stage=promotion_stage,
    archive_existing_versions=True
)