"""Streamlit app for manually testing the persisted baseline model.
Calls src.models.predict.predict_label directly (in-process, same
function the FastAPI endpoint uses) — see
specs/052-model-serving-api-ui/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from src.models.predict import load_artifacts, predict_label  # noqa: E402


@st.cache_resource
def get_model_and_vectorizer():
    return load_artifacts()


st.title("Hate Speech Detection")
st.write("Submit a comment to check whether the baseline model flags it as hateful.")

text = st.text_area("Comment", "")
language = st.selectbox("Language", ["en", "es"])

if st.button("Predict") and text.strip():
    model, vectorizer = get_model_and_vectorizer()
    result = predict_label(text, model, vectorizer, language)

    if result["label"] == 1:
        st.error("Prediction: HATE")
    else:
        st.success("Prediction: NOT HATE")

    if result["probability"] is not None:
        st.write(f"Confidence: {result['probability']:.1%}")
