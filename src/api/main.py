"""FastAPI endpoint exposing the persisted baseline model for hate
speech prediction — see specs/052-model-serving-api-ui/.
"""
from functools import lru_cache
from typing import Callable, Literal

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from src.models.predict import load_artifacts, predict_label

app = FastAPI(title="Hate Speech Detection API")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: Literal["en", "es"] = "en"


class PredictResponse(BaseModel):
    label: int
    prediction: str
    probability: float | None


@lru_cache
def _load_predictor() -> Callable[[str, str], dict]:
    """Load the model/vectorizer once per process and return a closure
    over them, so repeated requests don't touch disk again."""
    model, vectorizer = load_artifacts()
    return lambda text, language: predict_label(text, model, vectorizer, language)


def get_predictor() -> Callable[[str, str], dict]:
    """FastAPI dependency — overridden in tests via
    app.dependency_overrides to avoid loading the real (gitignored,
    CI-absent) .joblib artifacts."""
    return _load_predictor()


@app.get("/health")
def health() -> dict:
    """Liveness check — confirms the process is up and the model loaded."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, predictor=Depends(get_predictor)) -> dict:
    """Classify one comment as hate/not-hate. See contracts/predict.md
    (specs/052-model-serving-api-ui/) for the request/response shape."""
    return predictor(request.text, request.language)
