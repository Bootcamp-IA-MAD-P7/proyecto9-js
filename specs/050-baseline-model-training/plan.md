# Implementation Plan: Baseline hate speech classification model

**Branch**: `feature/baseline-model-training` | **Date**: 2026-09-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/050-baseline-model-training/spec.md`

## Summary

Split the augmented, preprocessed corpus with a stratified train/test
split, fit a TF-IDF vectorizer on the training split only, train Logistic
Regression, Linear SVM (`LinearSVC`), and Multinomial Naive Bayes on the
resulting features, compare them on held-out accuracy + F1 (hateful
class), and persist the winner with its matched vectorizer for reuse at
inference time by the serving task (issue #9).

## Technical Context

**Language/Version**: Python 3.14 (matches the rest of `src/`)

**Primary Dependencies**: scikit-learn (`LogisticRegression`, `LinearSVC`,
`MultinomialNB`, `train_test_split`, `f1_score`/`accuracy_score`), joblib
(persistence, already used by `scripts/build_features.py`), pandas

**Storage**: reads
`data/processed/enriched_comments_es_augmented_preprocessed.csv`; writes
`data/processed/baseline_model.joblib` and
`data/processed/baseline_vectorizer.joblib` (files, no database)

**Testing**: pytest, matching `tests/test_vectorize.py`'s style (small
in-memory fixtures, no dependency on the full CSV)

**Target Platform**: local CPU — Logistic Regression/LinearSVC/Naive Bayes
on ~18k sparse features are all far cheaper to train than the MarianMT
translation step; no GPU needed

**Project Type**: single Python library/CLI-script project (existing
`src/` + `scripts/` layout)

**Performance Goals**: fit + evaluate all three models on the full
121,478-row training split (80% of 151,848) in well under a few minutes
on CPU — linear models over sparse TF-IDF features are fast by
construction

**Constraints**: `MultinomialNB` requires non-negative input, which
TF-IDF already satisfies (unlike, say, mean-centered features), so no
extra preprocessing branch is needed to keep all three models on the same
feature matrix

**Scale/Scope**: 151,848 rows, bilingual (EN/ES), binary `label` target
(~37.3% positive per `specs/048-spanish-category-augmentation/spec.md`)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Practicality over precision**: three well-understood scikit-learn
  linear/NB baselines, no exotic architecture — pass.
- **Generalization over memorization**: stratified split + evaluating on
  a held-out test set (never seen during fit) is exactly the control this
  principle asks for; the full overfitting-gap check is issue #8, but
  this task's train/test separation is what makes that check possible
  later — pass.
- **Reproducibility**: scriptable (`scripts/train_baseline_model.py`),
  fixed `random_state` for the split and every stochastic estimator —
  pass.
- **Incremental delivery**: scoped to training + picking a winner only,
  no evaluation report/tuning/serving bundled in — pass.
- **Documented code**: short docstrings matching
  `src/features/vectorize.py`'s style — pass.
- No violations requiring the Complexity Tracking table.

## Project Structure

### Documentation (this feature)

```text
specs/050-baseline-model-training/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
└── spec.md               # Feature spec (already exists)
```

No `contracts/` directory: internal library step consumed by the next
task (evaluation, issue #8) via a plain function call / loading the
persisted `.joblib` files, same pattern as
`specs/049-classic-text-vectorization/plan.md`.

### Source Code (repository root)

```text
src/
└── models/
    ├── __init__.py         # NEW (empty, matching src/features/__init__.py)
    └── train.py             # NEW: split_dataset, build_models, train_and_compare

scripts/
└── train_baseline_model.py  # NEW: loads data, splits, fits vectorizer +
                              #      3 models, prints comparison, persists winner

tests/
└── test_train.py             # NEW: unit tests against a small fake corpus
```

**Structure Decision**: single-project layout, adding `src/models/`
alongside the existing `src/data/`, `src/preprocessing/`,
`src/features/` packages — the layout `specs/003-project-structure-setup/
spec.md` already reserved for this exact task.

## Complexity Tracking

No Constitution Check violations — table not needed.
