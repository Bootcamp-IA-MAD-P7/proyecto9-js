# Quickstart: Hyperparameter tuning with Optuna

## Run

```bash
python scripts/tune_hyperparameters.py
```

## Expected outcome

- Prints Optuna's per-trial progress (20 trials), then the best
  parameters and their mean CV F1 on the training split.
- Prints the final model's test-split metrics and a comparison against
  the un-tuned baseline's known test metrics.
- Writes `data/processed/tuned_model.joblib` and
  `data/processed/tuned_vectorizer.joblib` — does not touch
  `baseline_model.joblib`/`baseline_vectorizer.joblib` or
  `ensemble_model.joblib`/`ensemble_vectorizer.joblib`.

## Validate

```bash
python -m pytest tests/test_tuning.py -q
```
