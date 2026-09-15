# Spec 051: Model evaluation and overfitting control

## Linked issue

- GitHub issue: #8
- Kanban status at spec creation time: In Progress
- Level: Nivel Esencial

## What

Evaluate the persisted baseline model
(`specs/050-baseline-model-training/spec.md`) on both the training and
held-out test splits — accuracy, precision, recall, F1-score, and
confusion matrix — and flag the model as overfit if the train/test metric
gap exceeds 5 percentage points, per the project constitution's
"Generalization over memorization" principle.

## Why

`specs/001-hate-speech-detection/tasks.md` section 5 ("Evaluation") is
the next unblocked task now that a baseline model is trained and
persisted (`specs/050-baseline-model-training/spec.md`). That task only
reported test-set accuracy/F1 to pick a winner among three candidates; it
never looked at precision/recall individually, never produced a confusion
matrix, and never checked the model against its own training performance
— exactly the gap this task closes, and exactly the check the
constitution's 5-point overfitting threshold exists for.

## Scope

- `src/evaluation/evaluate.py`:
  - `compute_metrics(y_true, y_pred)`: returns accuracy, precision,
    recall, F1 (all on the hateful class, matching spec 050's tie-break
    metric) and the confusion matrix as a plain dict/array.
  - `check_overfitting(train_metrics, test_metrics, threshold=0.05)`:
    returns whether any of accuracy/precision/recall/F1 differs between
    train and test by more than `threshold`, and by how much.
- `scripts/evaluate_model.py`: reconstructs the exact same train/test
  split used by `scripts/train_baseline_model.py` (same
  `split_dataset`, same `random_state`), loads the persisted
  `data/processed/baseline_model.joblib` and
  `data/processed/baseline_vectorizer.joblib` (transform/predict only,
  never refit), computes metrics on both splits, prints a full report
  including the confusion matrix, and prints a clear overfit/not-overfit
  verdict.
- `tests/test_evaluate.py`: unit tests against small synthetic
  predictions with known metrics.

## EARS criteria

1. WHEN predictions and true labels are provided, THE SYSTEM SHALL
   compute accuracy, precision, recall, F1-score (all on the hateful
   class), and a confusion matrix.
2. WHEN the persisted model and vectorizer are loaded, THE SYSTEM SHALL
   use them only to transform/predict — never refit — so the evaluation
   reflects the exact artifacts already persisted by
   `specs/050-baseline-model-training/spec.md`.
3. WHEN train and test metrics are compared, THE SYSTEM SHALL flag the
   model as overfit IF any metric's train/test gap exceeds 5 percentage
   points, per the constitution's threshold.
4. THE SYSTEM SHALL reconstruct train/test splits deterministically
   (same `random_state` as spec 050), so re-running evaluation without
   re-running training still evaluates on the correct, unseen test rows.
5. THE SYSTEM SHALL expose `compute_metrics` and `check_overfitting` as
   independently testable functions accepting plain arrays, so tests run
   against small synthetic predictions without needing the full
   151,848-row dataset or the persisted model artifacts.

## Result

Ran `scripts/evaluate_model.py` against the full augmented dataset
(151,848 rows) and the persisted baseline model
(`data/processed/baseline_model.joblib`, Logistic Regression, from
`specs/050-baseline-model-training/spec.md`).

| Metric | Train (121,478 rows) | Test (30,370 rows) | Gap (pp) |
|---|---|---|---|
| Accuracy | 82.26% | 79.14% | +3.12 |
| Precision (hateful) | 74.73% | 71.04% | +3.69 |
| Recall (hateful) | 79.20% | 74.38% | +4.83 |
| F1 (hateful) | 76.90% | 72.67% | +4.23 |

Test confusion matrix: `[[TN=15616, FP=3433], [FN=2901, TP=8420]]`.

**VERDICT: not overfit.** Every metric's train/test gap stays under the
constitution's 5-point threshold; the closest is recall at 4.83 points,
which is worth watching if the dataset or model changes later but does
not currently cross the line. The model over-predicts hate slightly more
often than it misses it (3,433 false positives vs. 2,901 false
negatives on the test split) — reasonable for a moderation use case,
where a false positive (flagged for review) is cheaper than a false
negative (hate that slips through).

## Out of scope

- Retraining or choosing a different model — that decision was already
  made in `specs/050-baseline-model-training/spec.md`; this task only
  evaluates what was persisted.
- Per-category (homophobia/racism/etc.) breakdown — a reasonable
  follow-up, but not named in issue #8's scope.
- Hyperparameter tuning in response to an overfit verdict — issue #14
  (Optuna), a separate later task.
