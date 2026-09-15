# Tasks: Ensemble model for hate speech detection

**Input**: Design documents from `/specs/054-ensemble-model/`

**Tests**: included.

**Organization**: single cohesive story (US1).

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [x] T001 Create `specs/054-ensemble-model/` spec, plan, research,
      quickstart (this session, prior steps)

## Phase 2: Foundational

None needed — reuses `src/models/train.py::build_models`,
`src/features/vectorize.py`, `src/evaluation/evaluate.py`.

## Phase 3: User Story 1 - Train, evaluate, and compare the ensemble against the baseline (Priority: P1) 🎯 MVP

**Goal**: a hard-voting ensemble of the three baseline algorithms,
evaluated with the same methodology as spec 051, honestly compared
against the persisted baseline's known metrics.

**Independent Test**: run `scripts/train_ensemble_model.py` and confirm
it prints the comparison and persists the two new artifacts; run
`pytest tests/test_ensemble.py`.

### Tests for User Story 1

- [x] T002 [P] [US1] Write `tests/test_ensemble.py`: assert
      `build_ensemble()` returns a `VotingClassifier` with `voting="hard"`
      wrapping exactly the three `build_models()` estimator types; fit it
      on a small fake TF-IDF-vectorized corpus and assert `.predict()`
      returns labels in `{0, 1}` for each row

### Implementation for User Story 1

- [x] T003 [US1] Implement `src/models/ensemble.py::build_ensemble()`
      (depends on T002 existing first, red before green)
- [x] T004 [US1] Implement `scripts/train_ensemble_model.py`: same
      load/split/vectorize-on-train-only steps as
      `scripts/train_baseline_model.py`, fit `build_ensemble()`,
      `compute_metrics`/`check_overfitting` on train+test, print the
      report and an explicit improved/regressed/tied comparison against
      the baseline's spec-051 numbers, persist to
      `ensemble_model.joblib`/`ensemble_vectorizer.joblib` (depends on
      T003)
- [x] T005 [US1] Run `scripts/train_ensemble_model.py` against the real
      dataset, capture the actual numbers, and fill in spec.md's Result
      section (depends on T004)

**Checkpoint**: US1 complete and independently testable.

## Phase 4: Polish

- [x] T006 (skipped: no Nivel Medio section in the master tasks.md) [P] Update `specs/001-hate-speech-detection/tasks.md` (add a
      Nivel Medio note referencing this spec, if that file tracks it —
      otherwise skip, master tasks.md is Nivel Esencial-scoped)
- [x] T007 [P] Add a short README mention of the ensemble comparison
- [x] T008 Run `pytest` (full suite), confirm no regressions
- [x] T009 Run `quickstart.md` end-to-end

## Dependencies

Setup → Foundational (none) → T002 → T003 → T004 → T005 → Polish (T006-T009).
