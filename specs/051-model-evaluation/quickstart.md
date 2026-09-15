# Quickstart: Model evaluation and overfitting control

## Prerequisites

- `data/processed/enriched_comments_es_augmented_preprocessed.csv`,
  `data/processed/baseline_model.joblib`, and
  `data/processed/baseline_vectorizer.joblib` all exist (see
  `specs/050-baseline-model-training/spec.md`).
- Base `requirements.txt` installed (scikit-learn/joblib already core
  dependencies).

## Run

```bash
python scripts/evaluate_model.py
```

## Expected outcome

- Prints train-split and test-split metrics (accuracy, precision,
  recall, F1 on the hateful class) side by side.
- Prints the test-split confusion matrix.
- Prints a per-metric train/test gap and a final overfit/not-overfit
  verdict against the 5-point threshold.
- Writes no files — pure report. Exit code 0, no exceptions.

## Validate

```bash
python -m pytest tests/test_evaluate.py -q
```

All tests should pass, covering:
- `compute_metrics()` matches hand-computed accuracy/precision/recall/F1
  and confusion matrix on a small synthetic prediction set
- `check_overfitting()` correctly flags a synthetic train/test metrics
  pair with a >5-point gap, and correctly does not flag a pair within
  the threshold
