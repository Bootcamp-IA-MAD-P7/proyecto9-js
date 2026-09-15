# Implementation Plan: Model evaluation and overfitting control

**Branch**: `feature/model-evaluation` | **Date**: 2026-09-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/051-model-evaluation/spec.md`

## Summary

Reconstruct the exact train/test split used to train the persisted
baseline model, load the persisted model + vectorizer (transform/predict
only), compute accuracy/precision/recall/F1/confusion-matrix on both
splits, and compare them against the constitution's 5-point overfitting
threshold — reporting a clear verdict.

## Technical Context

**Language/Version**: Python 3.14 (matches the rest of `src/`)

**Primary Dependencies**: scikit-learn (`precision_score`, `recall_score`,
`f1_score`, `accuracy_score`, `confusion_matrix`), joblib (loading the
persisted artifacts), pandas

**Storage**: reads
`data/processed/enriched_comments_es_augmented_preprocessed.csv`,
`data/processed/baseline_model.joblib`,
`data/processed/baseline_vectorizer.joblib`; writes nothing (a report
task — spec.md's Result section is where numbers get recorded, matching
specs 048-050)

**Testing**: pytest, small synthetic prediction arrays with hand-computed
expected metrics — no dependency on the full CSV or the persisted
`.joblib` artifacts, matching `tests/test_train.py`'s style

**Target Platform**: local CPU — computing metrics on ~30k predictions
is near-instant, no meaningful performance concern

**Project Type**: single Python library/CLI-script project (existing
`src/` + `scripts/` layout)

**Performance Goals**: full report (both splits, ~151,848 rows total)
in well under a minute — dominated by re-vectorizing text, not metric
computation

**Constraints**: the train/test split MUST be reconstructed with the
exact same `text_column`/`label_column`/`test_size`/`random_state` as
`scripts/train_baseline_model.py` (via the same `split_dataset` function
from `specs/050-baseline-model-training/spec.md`) — any drift would
silently evaluate on the wrong rows and invalidate the overfitting check

**Scale/Scope**: 151,848 rows, binary `label` target, single persisted
model (whichever spec 050 picked — Logistic Regression per its Result
section)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Practicality over precision**: reuses existing scikit-learn metrics
  functions, no custom metric math — pass.
- **Generalization over memorization**: this task exists specifically to
  enforce the constitution's own 5-point train/test gap rule — pass, by
  construction.
- **Reproducibility**: scriptable (`scripts/evaluate_model.py`),
  deterministic split reconstruction — pass.
- **Incremental delivery**: scoped to evaluation/reporting only, no
  retraining or tuning bundled in — pass.
- **Documented code**: short docstrings matching
  `src/models/train.py`'s style — pass.
- No violations requiring the Complexity Tracking table.

## Project Structure

### Documentation (this feature)

```text
specs/051-model-evaluation/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
└── spec.md               # Feature spec (already exists)
```

No `contracts/` directory: internal library step, same pattern as specs
049/050.

### Source Code (repository root)

```text
src/
└── evaluation/
    ├── __init__.py         # already exists (empty, per src/ scaffolding)
    └── evaluate.py          # NEW: compute_metrics, check_overfitting

scripts/
└── evaluate_model.py        # NEW: reconstructs split, loads persisted
                              #      artifacts, prints the full report

tests/
└── test_evaluate.py          # NEW: unit tests against synthetic predictions
```

**Structure Decision**: single-project layout, adding `src/evaluation/`
alongside the existing `src/data/`, `src/preprocessing/`,
`src/features/`, `src/models/` packages — the layout
`specs/003-project-structure-setup/spec.md` already reserved for this
exact task.

## Complexity Tracking

No Constitution Check violations — table not needed.
