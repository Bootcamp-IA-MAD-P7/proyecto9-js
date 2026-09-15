# Tasks: Hyperparameter tuning with Optuna

**Tests**: included. **Organization**: single cohesive story (US1).

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [x] T001 Create `specs/055-optuna-hyperparameter-tuning/` spec, plan,
      research, quickstart (this session, prior steps)

## Phase 2: Foundational

None needed — reuses `split_dataset`, `build_tfidf_vectorizer`,
`compute_metrics`.

## Phase 3: User Story 1 - Tune, then honestly evaluate once (Priority: P1) 🎯 MVP

**Goal**: best Logistic Regression hyperparameters found via CV on
training data only, final model evaluated exactly once on the untouched
test split, compared against the baseline.

**Independent Test**: run `scripts/tune_hyperparameters.py`, confirm it
prints best params/CV score/test metrics/comparison and persists the
two new artifacts; run `pytest tests/test_tuning.py`.

### Tests for User Story 1

- [x] T002 [P] [US1] Write `tests/test_tuning.py`: use
      `optuna.trial.FixedTrial({"C": 1.0, "penalty": "l2",
      "class_weight": None})` to assert `suggest_hyperparameters()`
      returns exactly those three keys with valid types/ranges; build a
      tiny fake TF-IDF-vectorized corpus (~10 rows) and assert
      `objective()` returns a float in `[0, 1]` given a `FixedTrial` and
      `cv=2`

### Implementation for User Story 1

- [x] T003 [US1] Implement `src/models/tuning.py`:
      `suggest_hyperparameters(trial)` (`C`: `trial.suggest_float("C",
      0.01, 100, log=True)`; `penalty`: `trial.suggest_categorical(...,
      ["l1", "l2"])`; `class_weight`:
      `trial.suggest_categorical(..., [None, "balanced"])`);
      `objective(trial, X_train, y_train, cv=3)` (builds
      `LogisticRegression(solver="liblinear", random_state=42,
      **suggest_hyperparameters(trial))`, scores mean hateful-class F1
      via `StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)`)
      (depends on T002, red before green)
- [x] T004 [US1] Implement `scripts/tune_hyperparameters.py`: load data,
      `split_dataset`, fit TF-IDF on training split only, run
      `optuna.create_study(direction="maximize")` /
      `study.optimize(lambda t: objective(t, X_train_vec, y_train),
      n_trials=20)`, retrain `LogisticRegression` with
      `study.best_params` on the full training split, evaluate once via
      `compute_metrics` on the test split, print best params/CV
      score/test metrics/comparison vs. the baseline's spec-051 numbers,
      persist to `tuned_model.joblib`/`tuned_vectorizer.joblib` (depends
      on T003)
- [x] T005 [US1] Run `scripts/tune_hyperparameters.py` against the real
      dataset, capture the actual best params/metrics, and fill in
      spec.md's Result section (depends on T004)

**Checkpoint**: US1 complete and independently testable.

## Phase 4: Polish

- [x] T006 [P] Add a short README mention of the tuning result
- [x] T007 Run `pytest` (full suite), confirm no regressions
- [x] T008 Run `quickstart.md` end-to-end

## Dependencies

Setup → Foundational (none) → T002 → T003 → T004 → T005 → Polish (T006-T008).
