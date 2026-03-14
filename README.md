# 🍔 Zomato Delivery Time Prediction

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green.svg)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange.svg)
![DagsHub](https://img.shields.io/badge/Repository-DagsHub-purple.svg)
![Docker](https://img.shields.io/badge/Deployment-Docker-blue.svg)
![DVC](https://img.shields.io/badge/Data-DVC-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

> An end-to-end machine learning pipeline to predict food delivery times on Zomato using **LightGBM**, with full experiment tracking via **MLflow** and **DagsHub**, reproducible pipelines via **DVC**, and containerized deployment via **Docker**.

---

##  Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Highlights](#-project-highlights)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [ML Pipeline](#-ml-pipeline)
- [LightGBM Model](#-lightgbm-model)
- [MLflow & DagsHub Tracking](#-mlflow--dagshub-tracking)
- [DVC Pipeline](#-dvc-pipeline)
- [Installation & Setup](#-installation--setup)
- [Running the Project](#-running-the-project)
- [Docker Deployment](#-docker-deployment)
- [API Usage](#-api-usage)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Problem Statement

Food delivery platforms like Zomato operate in highly dynamic environments where delivery time depends on numerous factors — distance, weather, traffic density, vehicle type, and delivery partner ratings. Accurately predicting delivery time:

- Improves **customer satisfaction** by setting realistic expectations
- Helps **restaurants optimize** order preparation timing
- Allows **platform operations** to allocate delivery partners efficiently

This project builds a regression model to predict delivery time (in minutes) given real-world features extracted from Zomato delivery data.

---

## ✨ Project Highlights

-  End-to-end ML pipeline: ingestion → preprocessing → training → evaluation → deployment
-  **LightGBM** for fast, high-accuracy gradient boosting
-  **MLflow** experiment tracking with metrics, parameters, and artifacts
-  **DagsHub** remote tracking server for collaborative model analysis
-  **DVC** for data versioning and reproducible pipeline stages
-  **Flask** REST API for real-time predictions
-  **Docker** containerization for portable deployment
-  **GitHub Actions** CI/CD workflow for automated testing and deployment
-  Structured logging, custom exception handling, and modular codebase

---

##  Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| ML Model | LightGBM |
| Experiment Tracking | MLflow, DagsHub |
| Data Versioning | DVC |
| Web Framework | Flask |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Config Management | PyYAML, python-dotenv |

---

##  Project Structure

```
zomato-delivery-time-prediction/
│
├── .dvc/                          # DVC internal configuration
├── .github/
│   └── workflows/                 # GitHub Actions CI/CD pipelines
├── .pytest_cache/                 # Pytest cache
├── .venv/                         # Python virtual environment
│
├── data/
│   ├── raw/                       # Raw ingested data (tracked by DVC)
│   ├── processed/                 # Cleaned and feature-engineered data
│   └── final/                     # Train/test splits ready for modeling
│
├── models/                        # Saved model artifacts (.pkl / .txt)
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb # LightGBM experiments & tuning
│
├── scripts/                       # Standalone utility scripts
│
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py          # Data loading and validation
│   ├── data_preprocessing.py      # Cleaning, encoding, scaling
│   ├── feature_engineering.py     # Feature creation and selection
│   ├── model_trainer.py           # LightGBM training with MLflow logging
│   ├── model_evaluation.py        # Metrics computation and logging
│   ├── prediction_pipeline.py     # Inference pipeline for API
│   ├── logger.py                  # Custom logging configuration
│   └── exception.py               # Custom exception handling
│
├── static/
│   ├── css/                       # Frontend stylesheets
│   └── js/                        # Frontend JavaScript
│
├── templates/
│   └── index.html                 # Flask prediction web UI
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
│
├── .dvcignore                     # DVC ignore rules
├── .env                           # Environment variables (not committed)
├── .gitignore
├── app.py                         # Flask application entry point
├── check_model.py                 # Model sanity check script
├── Dockerfile                     # Docker build instructions
├── dvc.lock                       # DVC pipeline lock file
├── dvc.yaml                       # DVC pipeline stage definitions
├── params.yaml                    # Hyperparameters and config values
├── requirements.txt               # Python dependencies
├── requirements_docker.txt        # Docker-specific dependencies
├── run_information.json           # Run metadata and results summary
└── wait_for_server.py             # Server health check utility
```

---

## 📊 Dataset

The dataset contains historical Zomato delivery records with the following key features:

| Feature | Description |
|---|---|
| `Delivery_person_Age` | Age of the delivery partner |
| `Delivery_person_Ratings` | Average ratings (1–6 scale) |
| `Restaurant_latitude/longitude` | Restaurant GPS coordinates |
| `Delivery_location_latitude/longitude` | Customer GPS coordinates |
| `Order_Date` | Date of the order |
| `Time_Ordered` | Timestamp when order was placed |
| `Time_Order_picked` | Timestamp when order was picked up |
| `Weather_conditions` | Weather at time of delivery |
| `Road_traffic_density` | Traffic level (Low / Medium / High / Jam) |
| `Vehicle_condition` | Condition score of delivery vehicle |
| `Type_of_order` | Food category (Snack, Meal, Drinks, etc.) |
| `Type_of_vehicle` | Vehicle type (Bike, Scooter, etc.) |
| `multiple_deliveries` | Number of simultaneous deliveries |
| `Festival` | Whether delivery was during a festival |
| `City` | City type (Metropolitan, Urban, Semi-Urban) |
| **`Time_taken(min)`** | **Target: Delivery time in minutes** |

---

## 🔄 ML Pipeline

The pipeline is fully orchestrated using **DVC** and consists of these stages:

```
data_ingestion
      ↓
data_preprocessing
      ↓
feature_engineering
      ↓
model_training  ←── params.yaml (LightGBM hyperparameters)
      ↓
model_evaluation
      ↓
model_registry (MLflow / DagsHub)
```

Each stage is defined in `dvc.yaml` with explicit dependencies, outputs, and parameters — ensuring full reproducibility.

---

##  LightGBM Model

**LightGBM** (Light Gradient Boosting Machine) was selected as the primary model for this task due to its:

-  **Speed**: Leaf-wise tree growth with histogram-based splitting — significantly faster than XGBoost on large datasets
-  **Accuracy**: State-of-the-art performance on tabular regression tasks
-  **Categorical Support**: Native handling of categorical features without one-hot encoding
-  **Low Memory Usage**: Efficient memory footprint suitable for production

### Hyperparameters (from `params.yaml`)

```yaml
model:
  objective: regression
  metric: rmse
  boosting_type: gbdt
  num_leaves: 63
  learning_rate: 0.05
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 5
  min_child_samples: 20
  n_estimators: 500
  early_stopping_rounds: 50
  random_state: 42
```

### Feature Engineering Highlights

- **Haversine distance** computed from GPS coordinates
- **Time features**: hour of order, day of week, is_weekend, time_to_pickup
- **Interaction features**: traffic × weather, distance × vehicle_condition
- **Label encoding** for ordinal categoricals (traffic density, city type)
- **Outlier clipping** on delivery time and ratings

---

## MLflow & DagsHub Tracking

All experiments are tracked remotely on **DagsHub** using **MLflow**, providing a centralized dashboard for comparing runs, metrics, and artifacts.

### Setup DagsHub Remote Tracking

Add the following to your `.env` file:

```env
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/zomato-delivery-time-prediction.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

### What Gets Logged Per Run

| Category | Details |
|---|---|
| **Parameters** | All LightGBM hyperparameters from `params.yaml` |
| **Metrics** | RMSE, MAE, R² (train and validation) |
| **Artifacts** | Trained model file, feature importance plot, residual plot |
| **Tags** | Git commit hash, dataset version, run timestamp |

### Viewing Experiments

```python
# src/model_trainer.py (excerpt)
import mlflow
import mlflow.lightgbm
from dotenv import load_dotenv
import os

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("zomato-delivery-prediction")

with mlflow.start_run(run_name="lgbm_baseline"):
    mlflow.log_params(params)
    
    model = lgb.train(
        params=lgb_params,
        train_set=train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2_score)
    
    mlflow.lightgbm.log_model(model, artifact_path="lgbm_model")
    mlflow.log_artifact("reports/feature_importance.png")
```

Visit your DagsHub repo → **Experiments** tab to see all tracked runs, compare metrics across hyperparameter sweeps, and download the best model artifact.

---

##  DVC Pipeline

DVC manages data versioning and pipeline reproducibility.

```bash
# Run the full pipeline
dvc repro

# Check pipeline status (what needs to rerun)
dvc status

# Push data to remote storage
dvc push

# Pull data from remote storage
dvc pull
```

Pipeline stages in `dvc.yaml`:

```yaml
stages:
  data_ingestion:
    cmd: python src/data_ingestion.py
    deps: [src/data_ingestion.py]
    outs: [data/raw/]

  preprocessing:
    cmd: python src/data_preprocessing.py
    deps: [src/data_preprocessing.py, data/raw/]
    outs: [data/processed/]

  feature_engineering:
    cmd: python src/feature_engineering.py
    deps: [src/feature_engineering.py, data/processed/]
    outs: [data/final/]

  training:
    cmd: python src/model_trainer.py
    deps: [src/model_trainer.py, data/final/]
    params: [params.yaml:]
    outs: [models/lgbm_model.pkl]
    metrics: [run_information.json]

  evaluation:
    cmd: python src/model_evaluation.py
    deps: [src/model_evaluation.py, models/lgbm_model.pkl]
    metrics: [run_information.json]
```

---

##  Installation & Setup

### Prerequisites

- Python 3.9+
- Git
- Docker (optional, for containerized deployment)
- DagsHub account (for remote MLflow tracking)

### 1. Clone the Repository

```bash
git clone https://github.com/quamrl-hoda/zomato-delivery-time-prediction.git
cd zomato-delivery-time-prediction
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/zomato-delivery-time-prediction.mlflow
MLFLOW_TRACKING_USERNAME=<your-dagshub-username>
MLFLOW_TRACKING_PASSWORD=<your-dagshub-token>
```

### 5. Pull Data with DVC

```bash
dvc pull
```

---

##  Running the Project

### Run Full ML Pipeline

```bash
dvc repro
```

### Train Model Only

```bash
python src/model_trainer.py
```

### Check Model Performance

```bash
python check_model.py
```

### Launch Flask App

```bash
python app.py
```

Visit `http://localhost:5000` to use the prediction UI.

### Run Tests

```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

### Build the Docker Image

```bash
docker build -t zomato-delivery-predictor .
```

### Run the Container

```bash
docker run -p 5000:5000 --env-file .env zomato-delivery-predictor
```

### Docker Compose (with wait-for-server)

```bash
docker-compose up --build
```

The `wait_for_server.py` script ensures the Flask server is fully ready before accepting requests — useful in orchestrated or CI environments.

---

##  API Usage

Once the Flask app is running, you can hit the prediction endpoint:

### POST `/predict`

**Request Body (JSON):**

```json
{
  "Delivery_person_Age": 29,
  "Delivery_person_Ratings": 4.8,
  "Restaurant_latitude": 22.745049,
  "Restaurant_longitude": 75.892471,
  "Delivery_location_latitude": 22.765049,
  "Delivery_location_longitude": 75.912471,
  "Weather_conditions": "Sunny",
  "Road_traffic_density": "High",
  "Vehicle_condition": 2,
  "Type_of_order": "Meal",
  "Type_of_vehicle": "motorcycle",
  "multiple_deliveries": 0,
  "Festival": "No",
  "City": "Metropolitian",
  "time_to_pickup_minutes": 8
}
```

**Response:**

```json
{
  "predicted_delivery_time_minutes": 27.4,
  "status": "success"
}
```

---

##  Results

| Metric | Train | Validation |
|---|---|---|
| RMSE | 4.21 min | 4.87 min |
| MAE | 3.18 min | 3.74 min |
| R² Score | 0.834 | 0.812 |

> Results logged in `run_information.json` and tracked on DagsHub MLflow dashboard.

### Key Feature Importances (LightGBM)

1. `distance_km` (Haversine)
2. `Road_traffic_density`
3. `time_to_pickup_minutes`
4. `Weather_conditions`
5. `Delivery_person_Ratings`
6. `multiple_deliveries`
7. `City`

---

##  Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please make sure all tests pass before submitting a PR:

```bash
pytest tests/ -v
```

---

##  License

This project is licensed under the  Apache License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with  by <a href="https://github.com/quamrl-hoda">Quamrul Hoda</a> | Cognefy
</p>