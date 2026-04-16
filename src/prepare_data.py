"""
prepare_data.py
---------------
Loads the Breast Cancer Wisconsin dataset, saves the raw CSV, and
produces train/test splits that subsequent pipeline stages consume.
All parameters are read from params.yaml so DVC can track changes.
"""

import os
import yaml
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ── load params ───────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

TEST_SIZE    = params["data"]["test_size"]
RANDOM_STATE = params["data"]["random_state"]

# ── directories ───────────────────────────────────────────────────────────────
os.makedirs("data/raw",       exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── load dataset ──────────────────────────────────────────────────────────────
data    = load_breast_cancer()
X       = pd.DataFrame(data.data,   columns=data.feature_names)
y       = pd.Series(data.target,    name="target")

# ── save raw CSV ──────────────────────────────────────────────────────────────
raw_df = X.copy()
raw_df["target"] = y
raw_df.to_csv("data/raw/breast_cancer.csv", index=False)
print(f"✅  Raw dataset saved  →  data/raw/breast_cancer.csv  ({len(raw_df)} rows)")

# ── train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv( "data/processed/X_test.csv",  index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv( "data/processed/y_test.csv",  index=False)

print(f"✅  Train split: {len(X_train)} samples  |  Test split: {len(X_test)} samples")
print(f"✅  Processed splits saved  →  data/processed/")
