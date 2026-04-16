"""
main.py
-------
FastAPI inference server for the Breast Cancer classification champion model.
Loads the champion model from models/champion_model.pkl at startup.
"""

import os
import json
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# ── app setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Breast Cancer Classifier API",
    description="MLOps Group Project — inference endpoint for the champion model",
    version="1.0.0",
)

MODEL_PATH   = os.getenv("MODEL_PATH",   "models/champion_model.pkl")
INFO_PATH    = os.getenv("INFO_PATH",    "models/champion_info.json")

# ── load model at startup ─────────────────────────────────────────────────────
@app.on_event("startup")
def load_model():
    global MODEL, CHAMPION_INFO
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Champion model not found at {MODEL_PATH}. "
            "Run the DVC pipeline first: dvc repro"
        )
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)

    CHAMPION_INFO = {}
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH) as f:
            CHAMPION_INFO = json.load(f)

    print(f"✅  Model loaded from {MODEL_PATH}")


# ── request / response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="30 numerical features from the Breast Cancer Wisconsin dataset",
        example=[1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01,
                 2.776e-01, 3.001e-01, 1.471e-01, 2.419e-01, 7.871e-02,
                 1.095e+00, 9.053e-01, 8.589e+00, 1.534e+02, 6.399e-03,
                 4.904e-02, 5.373e-02, 1.587e-02, 3.003e-02, 6.193e-03,
                 2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03, 1.622e-01,
                 6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01, 1.189e-01],
    )

class PredictResponse(BaseModel):
    prediction: int
    label: str
    probability_malignant: float
    probability_benign: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    champion_run: str


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "champion_run": CHAMPION_INFO.get("run_name", "unknown"),
    }


@app.get("/")
def root():
    return {"message": "Breast Cancer Classifier API — visit /docs for Swagger UI"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        X = np.array(request.features).reshape(1, -1)
        prediction = int(MODEL.predict(X)[0])
        probabilities = MODEL.predict_proba(X)[0]

        # Breast cancer dataset: 0 = malignant, 1 = benign
        label = "benign" if prediction == 1 else "malignant"

        return {
            "prediction":            prediction,
            "label":                 label,
            "probability_malignant": round(float(probabilities[0]), 4),
            "probability_benign":    round(float(probabilities[1]), 4),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    return CHAMPION_INFO
