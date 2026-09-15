# Tasks: Model evaluation and overfitting control

**Input**: Design documents from `/specs/051-model-evaluation/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: included — this repo tests every `src/` module.

**Organization**: single cohesive story (US1) — compute metrics on both
splits, check the overfitting gap, report.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [x] T001 Create `specs/051-model-evaluation/` spec, plan, research,
      data-model, quickstart (this session, prior steps)
- [x] T002 Confirm `src/evaluation/__init__.py` already exists (it does,
      from initial scaffolding) — no action needed

## Phase 2: Foundational

No new foundational infrastructure needed — reuses
`src/models/train.py::split_dataset` and the persisted artifacts from
`specs/050-baseline-model-training/spec.md`.

## Phase 3: User Story 1 - Report full metrics and flag overfitting (Priority: P1) 🎯 MVP

**Goal**: given the persisted baseline model/vectorizer and the
reconstructed train/test split, report accuracy/precision/recall/F1 and
confusion matrix on both splits, and flag overfitting if any metric's
train/test gap exceeds 5 points.

**Independent Test**: run `python scripts/evaluate_model.py` against the
real artifacts and confirm it prints both splits' metrics, the confusion
matrix, and a verdict; run `pytest tests/test_evaluate.py` against small
synthetic predictions with hand-computed expected values.

### Tests for User Story 1

- [x] T003 [P] [US1] Write `tests/test_evaluate.py`: synthetic
      `y_true`/`y_pred` arrays (~10-12 values) with hand-computed
      accuracy/precision/recall/F1/confusion matrix; assert
      `compute_metrics()` matches those hand-computed values exactly;
      construct one synthetic `train_metrics`/`test_metrics` pair with a
      >5-point gap on one metric and assert `check_overfitting()` returns
      `is_overfit=True` with that metric named in `flagged_metrics`;
      construct another pair within the threshold and assert
      `is_overfit=False` with an empty `flagged_metrics`

### Implementation for User Story 1

- [x] T004 [US1] Implement `src/evaluation/evaluate.py`:
      `compute_metrics(y_true, y_pred)` (returns
      `{"accuracy", "precision", "recall", "f1", "confusion_matrix"}`,
      all on `pos_label=1`); `check_overfitting(train_metrics,
      test_metrics, threshold=0.05)` (returns `{"gaps", "is_overfit",
      "flagged_metrics"}`) (depends on T003 existing first so the tests
      fail red before this lands)
- [x] T005 [US1] Implement `scripts/evaluate_model.py`: load the
      preprocessed CSV, call `split_dataset` (same args as
      `scripts/train_baseline_model.py`) to reconstruct train/test,
      load `data/processed/baseline_model.joblib` and
      `data/processed/baseline_vectorizer.joblib` via `joblib.load`,
      `.transform()` (never `.fit()`) both splits, `.predict()` both,
      call `compute_metrics` on each split and `check_overfitting` on
      the pair, print the full report and verdict (depends on T004)
- [x] T006 [US1] Run `scripts/evaluate_model.py` against the real
      persisted artifacts, capture the actual metrics/verdict, and fill
      in spec.md's Result section with them (depends on T005)

**Checkpoint**: User Story 1 (the whole feature) is functional and
independently testable.

## Phase 4: Polish & Cross-Cutting Concerns

- [x] T007 [P] Update `specs/001-hate-speech-detection/tasks.md` section 5
      checkboxes to `[x]` once T003-T006 are done
- [x] T008 [P] Add the `scripts/evaluate_model.py` usage snippet to
      README.md's pipeline section
- [x] T009 Run `pytest` (full suite) and confirm no regressions
- [x] T010 Run `quickstart.md` end-to-end as a final sanity check

## Dependencies & Execution Order

- Setup (T001-T002) → done already, no blockers
- Foundational: none needed, US1 can start immediately
- Within US1: T003 (tests) before T004 (implementation, red-first) before
  T005 (script, depends on T004) before T006 (real run, depends on T005)
- Polish (T007-T010) depends on US1 (T003-T006) being complete

## Implementation Strategy

### MVP First (User Story 1 Only)

1. T003: write `tests/test_evaluate.py` against synthetic data, confirm
   it fails (no `src/evaluation/evaluate.py` yet)
2. T004: implement `src/evaluation/evaluate.py`, confirm T003 passes
3. T005: implement `scripts/evaluate_model.py`
4. T006: run it for real, record the Result numbers/verdict in spec.md
5. **STOP and VALIDATE**: `pytest tests/test_evaluate.py -q` passes

### Incremental Delivery

This feature *is* the MVP — there is only one story. Polish (T007-T010)
wraps it up: update the master tasks.md, document it in README, run the
full test suite, and do a final quickstart pass before opening the PR.
