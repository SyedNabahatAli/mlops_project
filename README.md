# MLOps End-to-End Pipeline — Breast Cancer Classification

A complete MLOps pipeline implementing data versioning, experiment tracking, model deployment, container orchestration, and CI/CD automation.

## Problem Statement
Binary classification on the **Breast Cancer Wisconsin** dataset to predict whether a tumour is malignant or benign using 30 numerical features.

## Tech Stack
| Component | Tool |
|-----------|------|
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Model Training | scikit-learn |
| Inference API | FastAPI |
| Containerisation | Docker |
| Orchestration | Kubernetes (Minikube) |
| CI/CD | GitHub Actions |

## Project Structure
```
mlops_project/
├── data/
│   ├── raw/               # Raw dataset (tracked by DVC)
│   └── processed/         # Preprocessed splits
├── src/
│   ├── prepare_data.py    # Data preparation stage
│   ├── train_experiments.py  # Training & MLflow logging
│   └── select_champion.py # Champion model selection
├── api/
│   ├── main.py            # FastAPI inference server
│   └── Dockerfile         # Container definition
├── k8s/
│   ├── deployment.yaml    # Kubernetes deployment
│   └── service.yaml       # Kubernetes service
├── tests/
│   └── test_api.py        # API smoke tests
├── .github/workflows/
│   └── mlops_pipeline.yml # CI/CD pipeline
├── dvc.yaml               # DVC pipeline stages
├── params.yaml            # Pipeline parameters
└── requirements.txt       # Python dependencies
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialise DVC and run the pipeline
```bash
dvc init
dvc repro
```

### 3. Start MLflow UI
```bash
mlflow ui
```

### 4. Run the API locally
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Build and run Docker container
```bash
docker build -t breast-cancer-api ./api
docker run -p 8000:8000 breast-cancer-api
```

### 6. Deploy on Kubernetes
```bash
kubectl apply -f k8s/
```

## Models Trained
- Logistic Regression (3 variants)
- Random Forest (3 variants)
- Gradient Boosting (2 variants)
- SVM (2 variants)
- KNN (2 variants)
- AdaBoost (2 variants)

Champion model is selected based on **highest recall** (minimising false negatives is critical in cancer detection).
