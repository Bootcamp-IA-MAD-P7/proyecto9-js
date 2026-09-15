# Implementation Plan: Hyperparameter tuning with Optuna

**Branch**: `feature/optuna-hyperparameter-tuning` | **Date**: 2026-09-16 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/055-optuna-hyperparameter-tuning/spec.md`

## Summary

Optuna study over `C`/`penalty`/`class_weight` for Logistic Regression,
scored by cross-validated hateful-class F1 on the training split only;
retrain the best trial on the full training split and evaluate it once,
honestly, on the untouched test split.

## Technical Context

**Language/Version**: Python 3.14

**Primary Dependencies**: `optuna` (already in `requirements.txt`),
scikit-learn (`StratifiedKFold`, `cross_val_score` or manual CV loop),
reuses `src/models/train.py::split_dataset`,
`src/features/vectorize.py`, `src/evaluation/evaluate.py`

**Storage**: reads
`data/processed/enriched_comments_es_augmented_preprocessed.csv`; writes
`data/processed/tuned_model.joblib`, `data/processed/tuned_vectorizer.joblib`

**Testing**: pytest, small fake corpus, `n_trials`/`cv` kept tiny in
tests for speed (matching `tests/test_ensemble.py`'s style)

**Target Platform**: local CPU

**Project Type**: single Python library/CLI-script project

**Performance Goals**: full study (20 trials × 3-fold CV = 60 fits) plus
one final fit/eval, on the ~121k-row training split, should complete in
a few minutes on CPU — same order of magnitude as training the three
baseline models already was; `n_trials=20` chosen specifically to keep
this practical before Thursday, not because more trials wouldn't help

**Constraints**: no trial may see the test split (EARS criterion 2);
`solver="liblinear"` fixed so both `l1`/`l2` are valid without a solver
axis in the search space

**Scale/Scope**: same dataset/split as specs 050/051; 3-parameter search
space

## Constitution Check

- **Practicality over precision**: small, fast search space and trial
  count over an exhaustive grid — pass.
- **Generalization over memorization**: CV on training only, single
  honest test-set check at the end — this task exists to serve this
  principle — pass.
- **Reproducibility**: fixed `random_state` on the split, the CV
  splitter, and the Optuna sampler — pass.
- **Incremental delivery**: tuning only, no auto-swap of the served
  model — pass.
- No violations requiring the Complexity Tracking table.

## Project Structure

### Documentation (this feature)

```text
specs/055-optuna-hyperparameter-tuning/
├── plan.md
├── research.md
├── quickstart.md
└── spec.md
```

No `data-model.md`/`contracts/`: reuses specs 050/051's shapes.

### Source Code (repository root)

```text
src/models/tuning.py              # NEW: suggest_hyperparameters, objective
scripts/tune_hyperparameters.py    # NEW: study, retrain, evaluate, persist
tests/test_tuning.py                # NEW
```

**Structure Decision**: single-project layout, one new file in the
existing `src/models/` package.

## Complexity Tracking

No violations — table not needed.
