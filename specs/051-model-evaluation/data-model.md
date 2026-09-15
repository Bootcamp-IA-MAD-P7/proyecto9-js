# Phase 1 Data Model: Model evaluation and overfitting control

No persistent database entities, and this feature writes no new files —
purely a read + report task.

## Reconstructed train/test split (intermediate, in-memory only)

- Same shape as `specs/050-baseline-model-training/data-model.md`'s
  split: `X_train, X_test, y_train, y_test` from
  `split_dataset(df, test_size=0.2, random_state=42)` on
  `data/processed/enriched_comments_es_augmented_preprocessed.csv`

## Persisted artifacts (input, read-only)

- `data/processed/baseline_model.joblib`: the fitted winning classifier
  from spec 050 (Logistic Regression per its Result section)
- `data/processed/baseline_vectorizer.joblib`: its matched, training-fit
  TF-IDF vectorizer — used via `.transform()` only, never `.fit()`

## Metrics report (dict, per split)

- `accuracy`, `precision`, `recall`, `f1` (all computed with
  `pos_label=1`, i.e. the hateful class)
- `confusion_matrix`: 2×2 array (`[[TN, FP], [FN, TP]]`)

## Overfitting verdict

- `gaps`: dict of `{metric_name: train_value - test_value}` for each of
  accuracy/precision/recall/F1
- `is_overfit`: `True` if any `abs(gap) > 0.05`
- `flagged_metrics`: list of metric names that individually exceeded the
  threshold (empty if `is_overfit` is `False`)

## Comparison report (script output, not persisted as a file)

- Printed by `scripts/evaluate_model.py`: train metrics, test metrics,
  per-metric gap, confusion matrix, and the final verdict. The concrete
  numbers get copied into spec.md's Result section once the script has
  run, matching specs 048-050's pattern.
