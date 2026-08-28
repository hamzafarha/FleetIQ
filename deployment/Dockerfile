# Phase 9 (Deployment) stub — revisit once serving requirements are known
# (batch vs real-time, expected QPS, latency budget).
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY deployment/ deployment/
COPY models/ models/
COPY config/ config/

EXPOSE 8000
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
