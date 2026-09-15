# Spec 055: Hyperparameter tuning with Optuna

## Linked issue

- GitHub issue: #14
- Kanban status at spec creation time: In Progress
- Level: Nivel Medio

## What

Define a hyperparameter search space for Logistic Regression (the
served baseline, `specs/050-baseline-model-training/spec.md`) and run
automated Optuna tuning against cross-validated F1 on the *training*
split only, then do one final, honest evaluation of the best
configuration on the untouched held-out test split — recording the best
parameters and resulting metrics, and comparing against the
un-tuned baseline.

## Why

`specs/001-hate-speech-detection/tasks.md`'s Nivel Medio arc continues:
issue #11 (ensemble) didn't beat the baseline, so tuning the baseline's
own hyperparameters is the next lever, per
`specs/001-hate-speech-detection/plan.md`. Logistic Regression is the
tuning target (not the ensemble from spec 054, which underperformed and
isn't the served model) since it's what `specs/052-model-serving-api-ui/
spec.md` actually serves — tuning it directly improves what's live.

Tuning must not touch the test split at all, or the resulting "best"
model would be quietly overfit to it: the constitution's "Generalization
over memorization" principle, and the exact reason
`specs/051-model-evaluation/spec.md` exists, both apply here just as
much as to training itself.

## Scope

- `src/models/tuning.py`:
  - `suggest_hyperparameters(trial)`: given an Optuna `Trial`, suggests
    `C` (log-uniform, regularization strength), `penalty` (`l1`/`l2`),
    and `class_weight` (`None`/`"balanced"`) — the three
    Logistic-Regression knobs most likely to move a linear TF-IDF
    baseline, kept to three to match the constitution's "practicality
    over precision" principle (a small, fast search beats an exhaustive
    one no one has time to run before Thursday).
  - `objective(trial, X_train, y_train, cv=3)`: builds a
    `LogisticRegression` from `suggest_hyperparameters(trial)` (fixed
    `solver="liblinear"`, the solver that supports both `l1` and `l2`),
    scores it via `cv`-fold stratified cross-validation on **training
    data only**, returns mean F1 (hateful class) for Optuna to maximize.
- `scripts/tune_hyperparameters.py`: loads the data, reconstructs the
  same train/test split and training-only TF-IDF fit as
  `scripts/train_baseline_model.py`, runs an Optuna study against
  `objective` (training split only), retrains a final model with the
  best trial's parameters on the full training split, evaluates it
  **once** on the held-out test split via the existing
  `src/evaluation/evaluate.py::compute_metrics`, prints best
  params/CV score/test metrics and an explicit comparison against the
  un-tuned baseline's spec-051 test numbers, and persists the tuned
  model + matched vectorizer under their own filenames (never
  overwrites the served baseline).
- `tests/test_tuning.py`: unit tests against a small fake corpus and a
  fixed/small trial count.

## EARS criteria

1. WHEN a trial suggests hyperparameters, THE SYSTEM SHALL only vary
   `C`, `penalty`, and `class_weight` — all other `LogisticRegression`
   arguments stay fixed (matching `specs/050-baseline-model-training/
   spec.md`'s `random_state=42`, plus `solver="liblinear"` so both
   penalties are valid).
2. WHEN a trial is scored, THE SYSTEM SHALL cross-validate on the
   training split only — the held-out test split from
   `specs/050-baseline-model-training/spec.md`'s `split_dataset` call
   MUST NOT be seen by any trial.
3. WHEN tuning finishes, THE SYSTEM SHALL retrain exactly one final
   model (the best trial's parameters, on the full training split) and
   evaluate it exactly once on the test split — never select a
   configuration by peeking at test performance.
4. THE SYSTEM SHALL report the best hyperparameters, their
   cross-validated training score, and the final test-split metrics,
   compared explicitly against the un-tuned baseline's test metrics.
5. THE SYSTEM SHALL persist the tuned model and vectorizer under
   distinct filenames from both the baseline's and the ensemble's, so
   this task doesn't silently change what's served
   (mirrors `specs/054-ensemble-model/spec.md`'s EARS criterion 4).

## Result

Ran `scripts/tune_hyperparameters.py` against the full augmented
dataset (same 151,848-row split as the baseline), 20 trials, 3-fold CV
on the training split only.

**Best hyperparameters**: `C=1.917`, `penalty="l2"`, `class_weight="balanced"`
(mean CV F1 on training data: 0.7230).

| Metric | Tuned (test) | Baseline (test) | Delta | Verdict |
|---|---|---|---|---|
| Accuracy | 79.21% | 79.14% | +0.07pp | improved |
| Precision (hateful) | 70.98% | 71.04% | -0.06pp | regressed |
| Recall (hateful) | 74.81% | 74.38% | +0.43pp | improved |
| F1 (hateful) | 72.84% | 72.67% | +0.17pp | improved |

**Modest but real improvement.** Unlike the ensemble (spec 054), tuning
didn't trade one metric for a worse regression elsewhere — F1 (the
project's tie-break metric), accuracy, and recall all improved slightly,
with only a negligible 0.06pp precision dip. The winning configuration
(`class_weight="balanced"`, `C≈1.9`) is close to the baseline's own
defaults (`class_weight="balanced"`, `C=1.0` — `specs/050-baseline-
model-training/spec.md`), which is itself informative: the baseline's
untuned choices were already close to a local optimum for this feature
space, so the ceiling on pure hyperparameter tuning (without changing
features or algorithm) is low. `penalty="l1"` never won a trial, meaning
the extra sparsity it would induce over TF-IDF's already-pruned
(`min_df=5`) vocabulary didn't help.

**Aside**: this run surfaced that scikit-learn 1.8+ deprecates
`LogisticRegression(penalty=...)` in favor of `l1_ratio` (verified
functionally equivalent — same resulting coefficients — before trusting
these results, see `src/models/tuning.py`'s warning-suppression comment);
worth migrating to `l1_ratio` directly before `penalty` is removed in
scikit-learn 1.10.

**Recommendation**: the improvement is small enough (+0.17pp F1) that
it's a judgment call whether it's worth promoting over the current
baseline — persisted separately at `data/processed/tuned_model.joblib`
for review, the served baseline is unchanged.

## Out of scope

- Tuning the ensemble (spec 054) or a neural model (issue #15, Nivel
  Avanzado) — this task tunes the served baseline only.
- Nested cross-validation / repeated CV for a tighter confidence
  interval on the CV score — a single `cv`-fold pass is the practical
  choice given the Thursday deadline; the untouched final test-set
  check is what actually matters for generalization, not CV precision.
- Automatically swapping the served model to the tuned version even if
  it wins — same deliberate-follow-up discipline as spec 054.
