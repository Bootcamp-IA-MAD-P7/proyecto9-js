# Quickstart: Ensemble model

## Run

```bash
python scripts/train_ensemble_model.py
```

## Expected outcome

- Prints train/test metrics for the ensemble (accuracy, precision,
  recall, F1 on the hateful class, confusion matrix).
- Prints a side-by-side comparison against the baseline's known test
  metrics (79.14% / 71.04% / 74.38% / 72.67%), stating whether each
  metric improved, regressed, or tied.
- Writes `data/processed/ensemble_model.joblib` and
  `data/processed/ensemble_vectorizer.joblib` — does not touch
  `baseline_model.joblib`/`baseline_vectorizer.joblib`.

## Validate

```bash
python -m pytest tests/test_ensemble.py -q
```
