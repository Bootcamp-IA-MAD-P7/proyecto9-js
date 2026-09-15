# Quickstart: Baseline model training

## Prerequisites

- `data/processed/enriched_comments_es_augmented_preprocessed.csv` exists
  (see `specs/048-spanish-category-augmentation/spec.md`).
- Base `requirements.txt` installed (scikit-learn is already a core
  dependency; no `requirements-nlp.txt` needed here — no translation
  model involved).

## Run

```bash
python scripts/train_baseline_model.py
```

## Expected outcome

- Prints, for each of the three models: accuracy and F1 (hateful class)
  on the held-out test split.
- Prints which model won and why (highest hateful-class F1).
- Writes `data/processed/baseline_model.joblib` and
  `data/processed/baseline_vectorizer.joblib`.
- Exit code 0, no exceptions.

## Validate

```bash
python -m pytest tests/test_train.py -q
```

All tests should pass, covering:
- `split_dataset()` preserves class balance across train/test (within a
  small tolerance) and is reproducible given the same `random_state`
- `build_models()` returns three distinct, unfitted estimator instances
- `train_and_compare()` returns per-model metrics and correctly picks the
  highest hateful-class-F1 model as the winner on a small synthetic
  corpus where the winner is known by construction
