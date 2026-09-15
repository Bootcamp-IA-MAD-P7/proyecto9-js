# Phase 1 Data Model: Queryable interface for the model (API + UI)

No persistent database entities. The request/response shapes are the API
contract (see `contracts/predict.md`).

## Loaded artifacts (in-memory, loaded once per process)

- `model`: the persisted fitted classifier
  (`data/processed/baseline_model.joblib`)
- `vectorizer`: its matched, training-fit TF-IDF vectorizer
  (`data/processed/baseline_vectorizer.joblib`)
- Both loaded read-only via `load_artifacts()`; never refit or mutated

## Prediction request (API input)

- `text`: str, required, non-empty — the raw comment to classify
- `language`: `"en"` | `"es"`, optional, defaults to `"en"` — selects the
  preprocessing pipeline applied before vectorizing (see
  `contracts/predict.md` for why this matters)

## Prediction result

- `label`: int, `0` (not hate) or `1` (hate)
- `prediction`: str, `"not_hate"` or `"hate"` — human-readable mirror of
  `label`
- `probability`: float in `[0, 1]` or `None` — the model's confidence in
  the predicted label, when the underlying estimator exposes
  `predict_proba` (`None` otherwise, per research.md)
