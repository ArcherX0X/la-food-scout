import logging
import os
import pathlib
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = pathlib.Path(
    os.getenv("MODEL_PATH", pathlib.Path(__file__).parent.parent / "models" / "model.pkl")
)

_bundle: dict | None = None


def load_model() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _bundle = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    return _bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except FileNotFoundError as e:
        logger.warning("Startup model load failed: %s", e)
    yield


app = FastAPI(
    title="LA Food Scout API",
    description="Predicts whether an LA restaurant is likely high-rated (≥4.5★).",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    neighborhood: str = Field(..., description="LA neighborhood name")
    primary_type: str = Field(..., description="Place type (e.g. restaurant, cafe)")
    price_level: float = Field(..., ge=0, le=4, description="Price level 0-4")
    review_count: float = Field(..., ge=0, description="Number of reviews")


class PredictResponse(BaseModel):
    high_rated_probability: float
    predicted_high_rated: bool
    inputs: dict


class MetaResponse(BaseModel):
    neighborhoods: list[str]
    place_types: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model file not found")
    try:
        load_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "ready"}


@app.get("/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
    try:
        bundle = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return MetaResponse(
        neighborhoods=bundle["neighborhoods"],
        place_types=bundle["place_types"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        bundle = load_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    pipeline = bundle["pipeline"]
    numeric = bundle["numeric_features"]
    categorical = bundle["categorical_features"]

    row = {
        "price_level": req.price_level,
        "review_count_log": np.log1p(req.review_count),
        "neighborhood": req.neighborhood,
        "primary_type": req.primary_type,
    }
    X = pd.DataFrame([row])[numeric + categorical]

    prob = float(pipeline.predict_proba(X)[0, 1])
    predicted = prob >= 0.5

    logger.info(
        "Predict: %s / %s / price=%.0f / reviews=%.0f -> prob=%.3f",
        req.neighborhood, req.primary_type, req.price_level, req.review_count, prob,
    )

    return PredictResponse(
        high_rated_probability=round(prob, 4),
        predicted_high_rated=predicted,
        inputs=req.model_dump(),
    )
