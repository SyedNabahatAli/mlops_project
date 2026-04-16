"""
select_champion.py
------------------
Queries all MLflow experiments, ranks runs by recall (or the metric
specified in params.yaml), enforces the minimum-recall gate, serialises
the winning pipeline to models/champion_model.pkl, and writes a JSON
summary to models/champion_info.json.
"""

import os
import json
import yaml
import pickle
import mlflow
import mlflow.sklearn

# ── load params ───────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

METRIC     = params["selection"]["metric"]       # e.g. "recall"
MIN_RECALL = params["selection"]["min_recall"]   # e.g. 0.90

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.tracking.MlflowClient()

os.makedirs("models", exist_ok=True)

# ── collect all runs across all experiments ───────────────────────────────────
experiments = client.search_experiments()
all_runs = []

for exp in experiments:
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{METRIC} DESC"],
    )
    all_runs.extend(runs)

if not all_runs:
    raise RuntimeError("No MLflow runs found. Run train_experiments.py first.")

# ── sort globally by the selection metric ─────────────────────────────────────
all_runs.sort(key=lambda r: r.data.metrics.get(METRIC, 0), reverse=True)
best_run = all_runs[0]
best_metric_value = best_run.data.metrics.get(METRIC, 0)

print(f"\n{'='*60}")
print(f"  Champion selection  —  metric: {METRIC}")
print(f"{'='*60}")
print(f"  Best run      : {best_run.info.run_name}")
print(f"  Experiment    : {best_run.info.experiment_id}")
print(f"  {METRIC:12s}  : {best_metric_value:.4f}")
print(f"  All metrics   : {best_run.data.metrics}")
print(f"{'='*60}\n")

# ── gate check ────────────────────────────────────────────────────────────────
if best_metric_value < MIN_RECALL:
    raise ValueError(
        f"Champion recall {best_metric_value:.4f} is below the required "
        f"minimum of {MIN_RECALL}. Aborting."
    )

# ── load and save the champion model ─────────────────────────────────────────
model_uri = f"runs:/{best_run.info.run_id}/model"
champion   = mlflow.sklearn.load_model(model_uri)

with open("models/champion_model.pkl", "wb") as f:
    pickle.dump(champion, f)

# ── write summary JSON ────────────────────────────────────────────────────────
champion_info = {
    "run_id":      best_run.info.run_id,
    "run_name":    best_run.info.run_name,
    "experiment":  best_run.info.experiment_id,
    "metrics":     best_run.data.metrics,
    "params":      best_run.data.params,
    "model_uri":   model_uri,
    "selection_metric": METRIC,
}

with open("models/champion_info.json", "w") as f:
    json.dump(champion_info, f, indent=2)

print("✅  Champion model saved  →  models/champion_model.pkl")
print("✅  Champion info saved   →  models/champion_info.json")
