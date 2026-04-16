"""
test_api.py
-----------
Smoke tests for the FastAPI inference server.
Run with:  pytest tests/test_api.py -v
The API must be running at http://localhost:8000 before running these tests,
OR use the TestClient for in-process testing (no server needed).
"""

import pytest
from fastapi.testclient import TestClient
import sys, os

# Allow importing from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── sample input (first sample from breast_cancer dataset — malignant) ─────────
SAMPLE_MALIGNANT = [
    1.799e+01, 1.038e+01, 1.228e+02, 1.001e+03, 1.184e-01,
    2.776e-01, 3.001e-01, 1.471e-01, 2.419e-01, 7.871e-02,
    1.095e+00, 9.053e-01, 8.589e+00, 1.534e+02, 6.399e-03,
    4.904e-02, 5.373e-02, 1.587e-02, 3.003e-02, 6.193e-03,
    2.538e+01, 1.733e+01, 1.846e+02, 2.019e+03, 1.622e-01,
    6.656e-01, 7.119e-01, 2.654e-01, 4.601e-01, 1.189e-01,
]

SAMPLE_BENIGN = [
    1.142e+01, 2.038e+01, 7.758e+01, 3.861e+02, 1.425e-01,
    2.839e-01, 2.414e-01, 1.052e-01, 2.597e-01, 9.744e-02,
    4.956e-01, 1.156e+00, 3.445e+00, 2.723e+01, 9.110e-03,
    7.458e-02, 5.661e-02, 1.867e-02, 5.963e-02, 9.208e-03,
    1.491e+01, 2.650e+01, 9.887e+01, 5.671e+02, 2.098e-01,
    8.663e-01, 6.869e-01, 2.575e-01, 6.638e-01, 1.730e-01,
]


@pytest.fixture(scope="module")
def client():
    """Create a TestClient — model must be present at models/champion_model.pkl"""
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_malignant(client):
    response = client.post("/predict", json={"features": SAMPLE_MALIGNANT})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert data["label"] in ["malignant", "benign"]
    assert 0.0 <= data["probability_malignant"] <= 1.0
    assert 0.0 <= data["probability_benign"] <= 1.0
    # For the first sample we expect malignant
    assert data["label"] == "malignant"


def test_predict_benign(client):
    response = client.post("/predict", json={"features": SAMPLE_BENIGN})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["malignant", "benign"]


def test_predict_wrong_feature_count(client):
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 422   # Unprocessable Entity


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
