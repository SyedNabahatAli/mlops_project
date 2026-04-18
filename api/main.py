import os
import json
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(
    title="Breast Cancer Classifier API",
    version="1.0.0",
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/champion_model")
INFO_PATH  = os.getenv("INFO_PATH", "models/champion_info.json")

MODEL = None
CHAMPION_INFO = {}

@app.on_event("startup")
def load_model():
    global MODEL, CHAMPION_INFO

    # load MLflow model (NO pickle)
    MODEL = mlflow.sklearn.load_model(MODEL_PATH)

    # load metadata
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH) as f:
            CHAMPION_INFO = json.load(f)

    print(" Model loaded successfully")


class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=30, max_length=30)


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        X = np.array(request.features).reshape(1, -1)

        pred = int(MODEL.predict(X)[0])
        prob = MODEL.predict_proba(X)[0]

        return {
            "prediction": pred,
            "label": "benign" if pred == 1 else "malignant",
            "probability_malignant": float(prob[0]),
            "probability_benign": float(prob[1]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "champion_run": CHAMPION_INFO.get("run_name", "unknown"),
    }