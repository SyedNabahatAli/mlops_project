import os
import json
import yaml
import mlflow
import mlflow.sklearn
import shutil

# ── load params ───────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

METRIC     = params["selection"]["metric"]
MIN_RECALL = params["selection"]["min_recall"]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.tracking.MlflowClient()

os.makedirs("models", exist_ok=True)

# ── collect runs ──────────────────────────────────────────────────────────────
experiments = client.search_experiments()
all_runs = []

for exp in experiments:
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{METRIC} DESC"],
    )
    all_runs.extend(runs)

if not all_runs:
    raise RuntimeError("No MLflow runs found.")

# ── select best ───────────────────────────────────────────────────────────────
all_runs.sort(key=lambda r: r.data.metrics.get(METRIC, 0), reverse=True)
best_run = all_runs[0]
best_metric_value = best_run.data.metrics.get(METRIC, 0)

print("\n" + "="*60)
print(f"  Champion selection — metric: {METRIC}")
print("="*60)
print(f"  Best run   : {best_run.info.run_name}")
print(f"  Value      : {best_metric_value:.4f}")
print("="*60 + "\n")

# ── gate check ────────────────────────────────────────────────────────────────
if best_metric_value < MIN_RECALL:
    raise ValueError(
        f"Best {METRIC} {best_metric_value:.4f} < {MIN_RECALL}"
    )

# ── load model from MLflow ────────────────────────────────────────────────────
model_uri = f"runs:/{best_run.info.run_id}/model"
champion = mlflow.sklearn.load_model(model_uri)

# ── remove existing model ─────────────────────────
if os.path.exists("models/champion_model"):
    shutil.rmtree("models/champion_model")

# ── save model in MLflow-native format ────────────────────────────────────────
mlflow.sklearn.save_model(
    champion,
    path="models/champion_model"
)

# ── save metadata ─────────────────────────────────────────────────────────────
champion_info = {
    "run_id": best_run.info.run_id,
    "run_name": best_run.info.run_name,
    "experiment": best_run.info.experiment_id,
    "metrics": best_run.data.metrics,
    "params": best_run.data.params,
    "model_uri": model_uri,
    "selection_metric": METRIC,
}

with open("models/champion_info.json", "w") as f:
    json.dump(champion_info, f, indent=2)

print("Champion model saved → models/champion_model")
print("Champion info saved → models/champion_info.json")