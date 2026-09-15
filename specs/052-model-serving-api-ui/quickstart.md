# Quickstart: Queryable interface for the model (API + UI)

## Prerequisites

- `data/processed/baseline_model.joblib` and
  `data/processed/baseline_vectorizer.joblib` exist (see
  `specs/050-baseline-model-training/spec.md`).
- Base `requirements.txt` installed (FastAPI, uvicorn, Streamlit already
  listed).

## Run the API

```bash
uvicorn src.api.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "you are all disgusting"}'
```

Expected: a JSON body with `label`, `prediction`, `probability` (see
`contracts/predict.md`). `GET /health` should return `{"status": "ok"}`.

## Run the Streamlit app

```bash
streamlit run src/app/streamlit_app.py
```

Expected: a browser tab opens with a text box and a "Predict" button;
submitting text shows the predicted label (and probability, if
available) in the same page.

## Validate

```bash
python -m pytest tests/test_predict.py tests/test_api.py -q
```

All tests should pass using a small fake fitted model/vectorizer — no
dependency on the real `.joblib` artifacts.
