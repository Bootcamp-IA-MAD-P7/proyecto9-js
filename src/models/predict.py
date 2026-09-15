"""Shared prediction logic used by both the FastAPI endpoint
(src/api/main.py) and the Streamlit app (src/app/streamlit_app.py), so
neither re-implements inference — see specs/052-model-serving-api-ui/.
"""
from pathlib import Path
from typing import Any

import joblib

from src.preprocessing.pipeline import preprocess

DEFAULT_MODEL_PATH = Path("data/processed/baseline_model.joblib")
DEFAULT_VECTORIZER_PATH = Path("data/processed/baseline_vectorizer.joblib")


def load_artifacts(
    model_path: Path = DEFAULT_MODEL_PATH,
    vectorizer_path: Path = DEFAULT_VECTORIZER_PATH,
) -> tuple[Any, Any]:
    """Load the persisted, already-fitted model and vectorizer
    (read-only — never refit here)."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_label(
    text: str, model: Any, vectorizer: Any, language: str = "en"
) -> dict[str, Any]:
    """Classify one raw text string as hate/not-hate.

    Applies the same cleaning/tokenization/stemming pipeline
    (src/preprocessing/pipeline.py) the model was trained on before
    vectorizing — the vectorizer's vocabulary is stemmed tokens (e.g.
    "disgust", not "disgusting"), so skipping this step would silently
    mismatch every prediction against the training distribution.

    `probability` is None when `model` has no `predict_proba` (e.g. a
    LinearSVC baseline persisted from a future re-run of
    scripts/train_baseline_model.py) — degrades gracefully instead of
    raising.
    """
    clean_text = preprocess(text, language)
    vector = vectorizer.transform([clean_text])
    label = int(model.predict(vector)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        hate_index = classes.index(1)
        probability = float(model.predict_proba(vector)[0][hate_index])

    return {
        "label": label,
        "prediction": "hate" if label == 1 else "not_hate",
        "probability": probability,
    }
