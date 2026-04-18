FROM python:3.14

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY models/champion_model/ ./models/champion_model/
COPY models/champion_info.json ./models/champion_info.json

ENV MODEL_PATH=models/champion_model
ENV INFO_PATH=models/champion_info.json

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]