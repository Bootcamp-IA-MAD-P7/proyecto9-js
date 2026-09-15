# Tasks: Classic text vectorization (TF-IDF / Bag of Words)

**Input**: Design documents from `/specs/049-classic-text-vectorization/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: included — this repo tests every `src/` module (see
`tests/test_preprocessing.py`, `tests/test_spanish_augmentation.py`).

**Organization**: this feature is a single cohesive story (US1) — vectorize
the corpus, compare representations, persist the baseline — so it isn't
split into independent P1/P2/P3 stories the way a multi-surface feature
would be.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [x] T001 Create `specs/049-classic-text-vectorization/` spec, plan,
      research, data-model, quickstart (this session, prior steps)

## Phase 2: Foundational

No new foundational infrastructure needed — reuses the existing `src/`
package layout (`src/features/__init__.py` already exists) and the
already-built `data/processed/enriched_comments_es_augmented_preprocessed.csv`
from `specs/048-spanish-category-augmentation/spec.md`.

## Phase 3: User Story 1 - Compare TF-IDF vs. Bag of Words and pick a baseline (Priority: P1) 🎯 MVP

**Goal**: given the preprocessed, augmented corpus, produce comparable
TF-IDF and Bag-of-Words feature matrices, report their vocabulary
size/sparsity, and persist the chosen (TF-IDF) vectorizer for reuse at
inference time.

**Independent Test**: run `python scripts/build_features.py` against the
real dataset and confirm it prints both representations' sizes and writes
`data/processed/tfidf_vectorizer.joblib`; run
`pytest tests/test_vectorize.py` against a small fake corpus with no
dependency on the full CSV.

### Tests for User Story 1

- [x] T002 [P] [US1] Write `tests/test_vectorize.py`: fake 5-6 row
      `clean_comment` corpus; assert `build_tfidf_vectorizer()` and
      `build_bow_vectorizer()` return fitted vectorizers whose
      `.transform()` output is a `scipy.sparse.csr_matrix` with
      `shape[0] == len(corpus)`; assert both vectorizers, given the same
      `min_df`, agree on `vocabulary_` (same token set, since the only
      difference between TF-IDF and BoW is the cell values, not which
      tokens are kept); assert `fit_transform_corpus()` returns both the
      matrix and the fitted vectorizer, and that calling `.transform()`
      again on new text does not change `vocabulary_` (no refit)

### Implementation for User Story 1

- [x] T003 [US1] Implement `src/features/vectorize.py`:
      `build_tfidf_vectorizer(min_df=5)` and `build_bow_vectorizer(min_df=5)`
      wrapping `sklearn.feature_extraction.text.TfidfVectorizer`/
      `CountVectorizer` with a shared `token_pattern` (whitespace-split,
      matching `clean_comment`'s already-tokenized-and-joined format) and
      `min_df`; `fit_transform_corpus(df, vectorizer, text_column="clean_comment")`
      returning `(matrix, fitted_vectorizer)` (depends on T002 existing
      first so the tests fail red before this lands)
- [x] T004 [US1] Implement `scripts/build_features.py`: load
      `data/processed/enriched_comments_es_augmented_preprocessed.csv`,
      fit both vectorizers via `fit_transform_corpus`, print vocabulary
      size/`nnz`/sparsity for each (per data-model.md's Comparison
      report), and persist the TF-IDF vectorizer to
      `data/processed/tfidf_vectorizer.joblib` via `joblib.dump`
      (depends on T003)
- [x] T005 [US1] Run `scripts/build_features.py` against the real 151,848
      row dataset, capture the actual vocabulary-size/sparsity numbers,
      and fill in spec.md's Result section with them (depends on T004)

**Checkpoint**: User Story 1 (the whole feature) is functional and
independently testable.

## Phase 4: Polish & Cross-Cutting Concerns

- [x] T006 [P] Update `specs/001-hate-speech-detection/tasks.md` section 3
      checkboxes to `[x]` once T002-T005 are done
- [x] T007 [P] Add the `scripts/build_features.py` usage snippet to
      README.md's pipeline section, matching the style of the existing
      preprocessing/augmentation snippets
- [x] T008 Run `pytest` (full suite) and confirm no regressions
- [x] T009 Run `quickstart.md` end-to-end as a final sanity check

## Dependencies & Execution Order

- Setup (T001) → done already, no blockers
- Foundational: none needed, US1 can start immediately
- Within US1: T002 (tests) before T003 (implementation, tests should fail
  red first) before T004 (script, depends on T003's functions) before T005
  (real run, depends on T004 existing)
- Polish (T006-T009) depends on US1 (T002-T005) being complete

## Parallel Example: User Story 1

```bash
# T002 and T003 touch different files but T003's implementation is what
# makes T002 pass — write T002 first, run it (red), then implement T003.
# T006 and T007 (Polish, different files) can run in parallel once US1 is done.
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. T002: write `tests/test_vectorize.py` against a fake corpus, confirm
   it fails (no `src/features/vectorize.py` yet)
2. T003: implement `src/features/vectorize.py`, confirm T002 passes
3. T004: implement `scripts/build_features.py`
4. T005: run it for real, record the Result numbers in spec.md
5. **STOP and VALIDATE**: `pytest tests/test_vectorize.py -q` passes,
   `data/processed/tfidf_vectorizer.joblib` exists

### Incremental Delivery

This feature *is* the MVP — there is only one story. Polish (T006-T009)
wraps it up: update the master tasks.md, document it in README, run the
full test suite, and do a final quickstart pass before opening the PR.
