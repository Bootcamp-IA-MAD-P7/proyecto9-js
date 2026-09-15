# Implementation Plan: Ensemble model for hate speech detection

**Branch**: `feature/ensemble-model` | **Date**: 2026-09-16 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/054-ensemble-model/spec.md`

## Summary

Wrap the existing `build_models()` estimators in a hard-voting
`VotingClassifier`, train/evaluate it with the exact same methodology as
`scripts/train_baseline_model.py`/`scripts/evaluate_model.py`, and report
a direct comparison against the persisted baseline's known metrics.

## Technical Context

**Language/Version**: Python 3.14

**Primary Dependencies**: scikit-learn (`VotingClassifier`), reuses
`src/models/train.py`, `src/features/vectorize.py`,
`src/evaluation/evaluate.py` — no new dependencies

**Storage**: reads
`data/processed/enriched_comments_es_augmented_preprocessed.csv`; writes
`data/processed/ensemble_model.joblib`,
`data/processed/ensemble_vectorizer.joblib` (distinct from the baseline's
files, per spec.md EARS criterion 4)

**Testing**: pytest, small fake corpus, matching `tests/test_train.py`'s
style

**Target Platform**: local CPU — hard-voting over 3 already-cheap linear/
NB models adds negligible cost versus training them individually

**Project Type**: single Python library/CLI-script project

**Performance Goals**: full train+evaluate run in the same order of
magnitude as `scripts/train_baseline_model.py` (that script trains all
three models already; this just adds the `VotingClassifier` wrapper fit,
not new model types)

**Constraints**: hard voting only (no `predict_proba` requirement on
`LinearSVC`) — see research.md

**Scale/Scope**: same 151,848-row dataset, same 80/20 stratified split

## Constitution Check

- **Practicality over precision**: hard voting over existing models,
  no new algorithm family or calibration complexity — pass.
- **Generalization over memorization**: reuses the leakage-avoidance
  (train-only vectorizer fit) and overfit-check machinery already built
  — pass.
- **Reproducibility**: scriptable, fixed `random_state` — pass.
- **Incremental delivery**: scoped to train+compare+persist only, no
  auto-swap of the served model — pass.
- No violations requiring the Complexity Tracking table.

## Project Structure

### Documentation (this feature)

```text
specs/054-ensemble-model/
├── plan.md
├── research.md
├── quickstart.md
└── spec.md
```

No `data-model.md`/`contracts/`: no new entities or external interface
beyond what specs 050/051 already defined (this task reuses their
metrics/persistence shapes as-is).

### Source Code (repository root)

```text
src/models/ensemble.py         # NEW: build_ensemble()
scripts/train_ensemble_model.py # NEW: train, evaluate, compare, persist
tests/test_ensemble.py          # NEW
```

**Structure Decision**: single-project layout, adding one file to the
existing `src/models/` package.

## Complexity Tracking

No violations — table not needed.
