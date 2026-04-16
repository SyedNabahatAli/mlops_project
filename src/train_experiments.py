"""
train_experiments.py
--------------------
Trains multiple scikit-learn classifiers on the Breast Cancer Wisconsin dataset,
logging each run to its own MLflow experiment.  Metrics logged include accuracy,
precision, recall, F1, and ROC-AUC so the selection script can rank by recall.

Reads processed splits produced by prepare_data.py (DVC stage).
"""

import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ── load params ───────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

RANDOM_STATE = params["training"]["random_state"]

# ── experiment / run definitions ──────────────────────────────────────────────
EXPERIMENTS = [
    {
        "experiment_name": "LogisticRegression_Experiments",
        "runs": [
            {
                "run_name": "LR_C0.01",
                "model": LogisticRegression(C=0.01, max_iter=10_000, random_state=RANDOM_STATE),
                "tags": {"model_family": "linear"},
            },
            {
                "run_name": "LR_C1.0",
                "model": LogisticRegression(C=1.0, max_iter=10_000, random_state=RANDOM_STATE),
                "tags": {"model_family": "linear"},
            },
            {
                "run_name": "LR_C100",
                "model": LogisticRegression(C=100.0, max_iter=10_000, random_state=RANDOM_STATE),
                "tags": {"model_family": "linear"},
            },
        ],
    },
    {
        "experiment_name": "RandomForest_Experiments",
        "runs": [
            {
                "run_name": "RF_n50_depth5",
                "model": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_STATE),
                "tags": {"model_family": "ensemble_bagging"},
            },
            {
                "run_name": "RF_n100_depthNone",
                "model": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=RANDOM_STATE),
                "tags": {"model_family": "ensemble_bagging"},
            },
            {
                "run_name": "RF_n200_depth10",
                "model": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RANDOM_STATE),
                "tags": {"model_family": "ensemble_bagging"},
            },
        ],
    },
    {
        "experiment_name": "GradientBoosting_Experiments",
        "runs": [
            {
                "run_name": "GB_lr0.05_n100",
                "model": GradientBoostingClassifier(learning_rate=0.05, n_estimators=100, random_state=RANDOM_STATE),
                "tags": {"model_family": "ensemble_boosting"},
            },
            {
                "run_name": "GB_lr0.1_n200",
                "model": GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, random_state=RANDOM_STATE),
                "tags": {"model_family": "ensemble_boosting"},
            },
        ],
    },
    {
        "experiment_name": "SVM_Experiments",
        "runs": [
            {
                "run_name": "SVM_rbf_C1",
                "model": SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE),
                "tags": {"model_family": "kernel"},
            },
            {
                "run_name": "SVM_rbf_C10",
                "model": SVC(kernel="rbf", C=10.0, probability=True, random_state=RANDOM_STATE),
                "tags": {"model_family": "kernel"},
            },
        ],
    },
    {
        "experiment_name": "KNN_Experiments",
        "runs": [
            {
                "run_name": "KNN_k3",
                "model": KNeighborsClassifier(n_neighbors=3),
                "tags": {"model_family": "instance_based"},
            },
            {
                "run_name": "KNN_k7",
                "model": KNeighborsClassifier(n_neighbors=7),
                "tags": {"model_family": "instance_based"},
            },
        ],
    },
    {
        "experiment_name": "AdaBoost_Experiments",
        "runs": [
            {
                "run_name": "ADA_n50_lr1.0",
                "model": AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=RANDOM_STATE),
                "tags": {"model_family": "ensemble_boosting"},
            },
            {
                "run_name": "ADA_n100_lr0.5",
                "model": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE),
                "tags": {"model_family": "ensemble_boosting"},
            },
        ],
    },
]


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


def main():
    # ── load processed splits from DVC stage ─────────────────────────────────
    X_train = pd.read_csv("data/processed/X_train.csv").values
    X_test  = pd.read_csv("data/processed/X_test.csv").values
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test  = pd.read_csv("data/processed/y_test.csv").values.ravel()

    print(f"Dataset: breast_cancer  (from data/processed/)")
    print(f"  Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}\n")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    total_runs = 0
    for exp_cfg in EXPERIMENTS:
        exp_name = exp_cfg["experiment_name"]
        mlflow.set_experiment(exp_name)

        for run_cfg in exp_cfg["runs"]:
            base_model = run_cfg["model"]
            run_name   = run_cfg["run_name"]
            tags       = run_cfg.get("tags", {})

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    base_model),
            ])

            with mlflow.start_run(run_name=run_name, tags=tags):
                pipeline.fit(X_train, y_train)

                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1]

                metrics = compute_metrics(y_test, y_pred, y_prob)
                mlflow.log_metrics(metrics)

                params_model = base_model.get_params()
                mlflow.log_params({k: str(v) for k, v in params_model.items()})

                mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="model",
                    input_example=X_train[:3],
                )

                total_runs += 1
                print(
                    f"[{exp_name}] {run_name:30s}  "
                    f"recall={metrics['recall']:.4f}  "
                    f"roc_auc={metrics['roc_auc']:.4f}"
                )

    print(f"\n  Finished {total_runs} runs across {len(EXPERIMENTS)} experiments.")


if __name__ == "__main__":
    main()
