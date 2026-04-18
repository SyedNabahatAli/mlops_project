# Setup & Run Guide — MLOps End-to-End Pipeline

This guide walks through running the full pipeline from scratch on a local machine.

---

## Prerequisites

Make sure the following are installed:
- Python 3.14+
- Docker Desktop (with Kubernetes enabled, or Minikube)
- Git

---

## Step 1 — Clone the Repository & Install Dependencies

```bash
git clone https://github.com/<your-username>/mlops_project.git
cd mlops_project
create virtual env and activate it
pip install -r requirements.txt
```

---

## Step 2 — Initialise Git & DVC

```bash
git init
dvc init
git add .
git commit -m "Initial project setup"
```

---

## Step 3 — Start the MLflow Tracking Server

Open a **separate terminal** and run:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Leave this running. You can view the MLflow UI at: http://127.0.0.1:5000

---

## Step 4 — Run the Full DVC Pipeline

```bash
dvc repro
```

This runs 3 stages automatically:

| Stage | Script | Output |
|-------|--------|--------|
| `prepare` | `src/prepare_data.py` | `data/raw/breast_cancer.csv`, `data/processed/*.csv` |
| `train` | `src/train_experiments.py` | MLflow experiment runs (14 total) |
| `select` | `src/select_champion.py` | `models/champion_model`, `models/champion_info.json` |

To re-run only changed stages:
```bash
dvc repro   # DVC skips unchanged stages automatically
```

To force a full re-run:
```bash
dvc repro --force
```

---

## Step 5 — Inspect Experiments in MLflow UI

Open http://127.0.0.1:5000 in your browser.

You will see 6 experiments:
- LogisticRegression_Experiments
- RandomForest_Experiments
- GradientBoosting_Experiments
- SVM_Experiments
- KNN_Experiments
- AdaBoost_Experiments

The champion model (highest recall) is saved to `models/champion_model`.

---

## Step 6 — Run the API Locally (without Docker)

```bash
uvicorn api.main:app --reload --port 8000
```

Test it:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/info
```

Swagger UI: http://localhost:8000/docs

---

## Step 7 — Run Tests

```bash
pytest tests/test_api.py -v
```

---

## Step 8 — Build & Run the Docker Container

> Make sure the DVC pipeline has been run first so `models/champion_model` exists.

```bash
# Build the image (run from project root)
docker build -t breast-cancer-api:latest -f api/Dockerfile .

# Run the container
docker run -p 8000:8000 breast-cancer-api:latest
```

Test the containerised API:
```bash
curl http://localhost:8000/health
```

---

## Step 9 — Deploy on Kubernetes (Minikube)

### Option A: Docker Desktop Kubernetes
Make sure Kubernetes is enabled in Docker Desktop settings.
## once minikube is enabled on desktop, you can proceed to test by running this on terminal:
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

### Option B: Minikube
```bash
minikube start
# Load the local Docker image into Minikube
minikube image load breast-cancer-api:latest
```

### Deploy
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check pods are running
kubectl get pods
kubectl get services
```

### Access the service

**Docker Desktop:**
```bash
curl http://localhost:30800/health
```

**Minikube:**
```bash
minikube service breast-cancer-api-service --url
# Then use the URL printed to curl the /health endpoint
```

---

## Step 10 — CI/CD with GitHub Actions

### Setup secrets in GitHub
Go to your repo → Settings → Secrets and variables → Actions, and add:

| Secret | Value |
|--------|-------|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_PASSWORD` | Your Docker Hub password or access token |

### Push to trigger the pipeline
```bash
git add .
git commit -m "feat: complete MLOps pipeline"
git push origin main
```

The pipeline will:
1. Run `dvc repro` (prepare → train → select champion)
2. Run pytest API tests
3. Build and push the Docker image to Docker Hub

---

## Useful Commands Reference

```bash
# Check DVC pipeline status
dvc status

# View DVC pipeline DAG
dvc dag

# View MLflow experiments
mlflow experiments list

# Check Kubernetes pods
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>

# Delete Kubernetes deployment
kubectl delete -f k8s/
```
