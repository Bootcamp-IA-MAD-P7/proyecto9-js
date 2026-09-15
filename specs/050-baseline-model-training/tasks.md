# Tasks: Baseline hate speech classification model

**Input**: Design documents from `/specs/050-baseline-model-training/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: included — this repo tests every `src/` module.

**Organization**: single cohesive story (US1) — split, train three
models, compare, persist the winner.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [x] T001 Create `specs/050-baseline-model-training/` spec, plan,
      research, data-model, quickstart (this session, prior steps)
- [x] T002 Create `src/models/__init__.py` (empty, matches
      `src/features/__init__.py`)

## Phase 2: Foundational

No new foundational infrastructure needed — reuses the existing `src/`
package layout and
`data/processed/enriched_comments_es_augmented_preprocessed.csv` from
`specs/048-spanish-category-augmentation/spec.md`.

## Phase 3: User Story 1 - Train and compare three baselines, persist the winner (Priority: P1) 🎯 MVP

**Goal**: given the preprocessed, augmented corpus, produce a stratified
train/test split, a training-only-fit TF-IDF vectorizer, three trained
classifiers with comparable held-out metrics, and persist the
best-performing model + its matched vectorizer.

**Independent Test**: run `python scripts/train_baseline_model.py`
against the real dataset and confirm it prints all three models'
accuracy/F1 and writes both `.joblib` artifacts; run
`pytest tests/test_train.py` against a small synthetic corpus with a
known-in-advance winner.

### Tests for User Story 1

- [x] T003 [P] [US1] Write `tests/test_train.py`: small fake DataFrame
      (~20 rows, imbalanced ~similar to the real 37/63 split); assert
      `split_dataset()` returns train/test splits whose `label` mean is
      within a small tolerance of the full set's (stratification check)
      and is identical across two calls with the same `random_state`;
      assert `build_models()` returns 3 distinct unfitted estimator
      instances (`LogisticRegression`, `LinearSVC`, `MultinomialNB`);
      assert `train_and_compare()` returns a dict/DataFrame with
      `accuracy`/`f1_hateful` per model name and a `winner` matching the
      model with the highest `f1_hateful` on a synthetic corpus
      constructed so the winner is unambiguous

### Implementation for User Story 1

- [x] T004 [US1] Implement `src/models/train.py`:
      `split_dataset(df, test_size=0.2, random_state=42)` (stratified on
      `label`, returns `X_train, X_test, y_train, y_test`);
      `build_models()` (returns
      `{"logistic_regression": LogisticRegression(class_weight="balanced"),
      "linear_svm": LinearSVC(class_weight="balanced"),
      "naive_bayes": MultinomialNB()}`); `train_and_compare(models, X_train_vec,
      y_train, X_test_vec, y_test)` (fits each, scores accuracy + F1 on
      `label==1`, returns per-model metrics and the winning model name)
      (depends on T003 existing first so the tests fail red before this
      lands)
- [x] T005 [US1] Implement `scripts/train_baseline_model.py`: load
      `data/processed/enriched_comments_es_augmented_preprocessed.csv`,
      call `split_dataset`, fit a fresh `TfidfVectorizer` on `X_train`
      only (per research.md — do not reuse
      `data/processed/tfidf_vectorizer.joblib`), transform both splits,
      call `train_and_compare`, print the comparison table and the
      winner, persist the winning fitted model to
      `data/processed/baseline_model.joblib` and the training-fit
      vectorizer to `data/processed/baseline_vectorizer.joblib` (depends
      on T004)
- [x] T006 [US1] Run `scripts/train_baseline_model.py` against the real
      151,848-row dataset, capture the actual per-model metrics and the
      winner, and fill in spec.md's Result section with them (depends on
      T005)

**Checkpoint**: User Story 1 (the whole feature) is functional and
independently testable.

## Phase 4: Polish & Cross-Cutting Concerns

- [x] T007 [P] Update `specs/001-hate-speech-detection/tasks.md` section 4
      checkboxes to `[x]` once T003-T006 are done
- [x] T008 [P] Add the `scripts/train_baseline_model.py` usage snippet to
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

1. T003: write `tests/test_train.py` against a fake corpus, confirm it
   fails (no `src/models/train.py` yet)
2. T004: implement `src/models/train.py`, confirm T003 passes
3. T005: implement `scripts/train_baseline_model.py`
4. T006: run it for real, record the Result numbers in spec.md
5. **STOP and VALIDATE**: `pytest tests/test_train.py -q` passes, both
   `.joblib` artifacts exist

### Incremental Delivery

This feature *is* the MVP — there is only one story. Polish (T007-T010)
wraps it up: update the master tasks.md, document it in README, run the
full test suite, and do a final quickstart pass before opening the PR.
