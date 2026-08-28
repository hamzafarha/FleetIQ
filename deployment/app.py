"""Phase 9 (Deployment) stub. Not wired to a real model yet — placeholder
so the repo shape reflects the full pipeline from day one.

Run with: uvicorn deployment.app:app --reload
"""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Smart ETA Prediction API")


class TripRequest(BaseModel):
    pickup_latitude: float
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float
    pickup_datetime: str


class TripResponse(BaseModel):
    predicted_duration_minutes: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=TripResponse)
def predict(trip: TripRequest):
    # TODO (Phase 9): load trained model artifact from models/ and replace
    # this placeholder once a model exists.
    raise NotImplementedError("No trained model wired in yet — Phase 6+ not complete.")
