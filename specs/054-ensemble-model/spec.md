# Spec 054: Ensemble model for hate speech detection

## Linked issue

- GitHub issue: #11
- Kanban status at spec creation time: In Progress
- Level: Nivel Medio

## What

Train a hard-voting ensemble over the same three baseline algorithms
(`specs/050-baseline-model-training/spec.md`: Logistic Regression,
Linear SVM, Multinomial Naive Bayes) and compare it against the
persisted single-model baseline on the identical held-out test split,
using the same metrics as `specs/051-model-evaluation/spec.md`
(accuracy, precision, recall, F1 on the hateful class, confusion
matrix), reporting honestly whether it actually improves.

## Why

`specs/001-hate-speech-detection/tasks.md`'s Nivel Medio scope (the
issue #11-#14 arc) starts here. The baseline (Logistic Regression,
79.14% accuracy / 72.67% F1 on the hateful class, confirmed not overfit)
is the bar to beat. A voting ensemble of the three already-trained
algorithm types is the natural next step: it reuses
`src/models/train.py::build_models()` instead of introducing a new
algorithm family, and answers a concrete question — does combining these
three do better than the best of them alone — before investing further
effort in hyperparameter tuning (issue #14) or a YouTube-facing feature
(issue #12) on top of whichever model is actually best.

## Scope

- `src/models/ensemble.py`: `build_ensemble()` — a scikit-learn
  `VotingClassifier` wrapping the three estimators from
  `build_models()`, using **hard voting** (majority vote across the
  three predicted labels), not soft voting — see research.md for why.
- `scripts/train_ensemble_model.py`: mirrors
  `scripts/train_baseline_model.py`'s methodology exactly (same
  `split_dataset` call, same training-split-only TF-IDF fit, per
  `specs/050-baseline-model-training/spec.md`'s leakage-avoidance rule),
  fits the ensemble, computes train/test metrics via the existing
  `src/evaluation/evaluate.py::compute_metrics`/`check_overfitting`,
  prints a side-by-side comparison against the baseline's known test
  metrics, and persists the ensemble + its matched vectorizer under
  their own filenames (does not overwrite/replace the served baseline
  artifacts from spec 052).
- `tests/test_ensemble.py`: unit tests against a small fake corpus.

## EARS criteria

1. WHEN the ensemble is built, THE SYSTEM SHALL combine exactly the
   three algorithm types already used for the baseline comparison
   (`specs/050-baseline-model-training/spec.md`), via hard-voting
   majority rule.
2. WHEN the ensemble is evaluated, THE SYSTEM SHALL use the identical
   train/test split and TF-IDF-fit-on-training-only methodology as the
   baseline, so the comparison is apples-to-apples.
3. THE SYSTEM SHALL report accuracy, precision, recall, F1 (hateful
   class), and confusion matrix for the ensemble on the test split, and
   explicitly state whether each metric improved, regressed, or tied
   versus the persisted baseline's spec-051 numbers.
4. THE SYSTEM SHALL persist the ensemble model and its matched
   vectorizer under distinct filenames from the baseline's
   (`data/processed/baseline_model.joblib`,
   `baseline_vectorizer.joblib`), so this task does not silently change
   what `specs/052-model-serving-api-ui/spec.md`'s API/UI serve.

## Result

Ran `scripts/train_ensemble_model.py` against the full augmented dataset
(same 151,848-row split as the baseline).

| Metric | Ensemble (test) | Baseline (test) | Delta | Verdict |
|---|---|---|---|---|
| Accuracy | 79.45% | 79.14% | +0.31pp | improved |
| Precision (hateful) | 72.54% | 71.04% | +1.50pp | improved |
| Recall (hateful) | 72.19% | 74.38% | -2.19pp | **regressed** |
| F1 (hateful) | 72.37% | 72.67% | -0.30pp | **regressed** |

**The ensemble does not clearly beat the baseline.** It trades a small
accuracy/precision gain for a larger recall loss, and F1 — the tie-break
metric used throughout this project (specs 050/051) — comes out
slightly worse. Worse still, the ensemble's own train/test gap analysis
flags it as **overfit** on two metrics (recall: 6.31pp gap, F1: 5.62pp
gap — both over the constitution's 5-point threshold), unlike the
baseline, which passed clean on all four metrics
(`specs/051-model-evaluation/spec.md`). Hard-voting over three linear/NB
models trained on the same features doesn't diversify their errors
enough to help, and majority voting biases toward the two more
accuracy-leaning models (Logistic Regression, Linear SVM) at the expense
of Naive Bayes' comparatively higher recall.

**Recommendation**: keep serving the Logistic Regression baseline
(unchanged — this task never touched
`baseline_model.joblib`/`baseline_vectorizer.joblib`, per EARS
criterion 4). The ensemble is persisted at
`data/processed/ensemble_model.joblib` for reference, but is not a
recommended upgrade as implemented. A stacking meta-learner or soft
voting with calibrated probabilities (both explicitly out of scope here)
would be the next things to try if ensembling is revisited.

## Out of scope

- Automatically swapping the served model (API/Streamlit) to the
  ensemble even if it wins — a deliberate follow-up decision, not
  silently done here (mirrors spec 052's read-only-artifact discipline).
- Stacking (a meta-learner over the three base models' outputs) — hard
  voting is the simpler, cheaper technique named first in issue #11;
  stacking is a reasonable future iteration if hard voting underwhelms,
  not required by this spec.
- Soft voting / probability calibration — see research.md; would add a
  `CalibratedClassifierCV` wrapper around `LinearSVC` purely to enable
  probability averaging, a real complexity cost the constitution's
  "practicality over precision" principle argues against for a first
  ensemble pass.
