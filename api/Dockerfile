# ── base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── install dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    scikit-learn==1.4.2 \
    numpy==1.26.4 \
    pydantic==2.7.1

# ── copy application code ─────────────────────────────────────────────────────
COPY api/main.py ./api/main.py

# ── copy champion model artifacts ─────────────────────────────────────────────
# The champion model and info are built by the DVC pipeline before Docker build
COPY models/champion_model.pkl ./models/champion_model.pkl
COPY models/champion_info.json ./models/champion_info.json

# ── environment variables ─────────────────────────────────────────────────────
ENV MODEL_PATH=models/champion_model.pkl
ENV INFO_PATH=models/champion_info.json

# ── expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── run the server ────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
