# Implementation Plan: Classic text vectorization (TF-IDF / Bag of Words)

**Branch**: `feature/classic-text-vectorization` | **Date**: 2026-09-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/049-classic-text-vectorization/spec.md`

## Summary

Vectorize the `clean_comment` column of the Spanish-augmented, preprocessed
dataset with both `TfidfVectorizer` and `CountVectorizer` (scikit-learn),
sharing the same tokenizer/`min_df` configuration so the two feature spaces
are comparable, report vocabulary size and sparsity for each, and record
TF-IDF as the baseline representation feeding model training (issue #7) —
standard choice for a sparse linear-model baseline (Logistic
Regression/Linear SVM per `specs/001-hate-speech-detection/tasks.md`
section 4), since it down-weights corpus-common tokens surviving
preprocessing (e.g. `url`) automatically instead of requiring a separate
stopword pass tuned for this exact vocabulary.

## Technical Context

**Language/Version**: Python 3.14 (matches the rest of `src/`)

**Primary Dependencies**: scikit-learn (`TfidfVectorizer`, `CountVectorizer`,
already a project dependency for the eventual model-training task),
pandas, scipy (sparse matrices, transitive via scikit-learn)

**Storage**: reads
`data/processed/enriched_comments_es_augmented_preprocessed.csv` (files,
no database); fitted vectorizers persisted as `.joblib` files under
`data/processed/` for reuse at inference time (per tasks.md section 4:
"persist it alongside its fitted vectorizer")

**Testing**: pytest, matching `tests/test_preprocessing.py` and
`tests/test_spanish_augmentation.py`'s existing style (small in-memory
fixtures, no dependency on the full CSV)

**Target Platform**: local CPU (same as the rest of the data pipeline;
scikit-learn vectorization is far cheaper than the MarianMT translation
step and needs no GPU)

**Project Type**: single Python library/CLI-script project (existing
`src/` + `scripts/` layout)

**Performance Goals**: fit-transform the full 151,848-row corpus in well
under a minute (in line with preprocessing's 34.4s for the same row count
— vectorization is cheaper per row than the stemming pipeline it follows)

**Constraints**: feature matrices must stay sparse (scipy `csr_matrix`) —
a dense 151,848 × vocab_size matrix would be memory-prohibitive at
realistic vocabulary sizes (tens of thousands of tokens)

**Scale/Scope**: 151,848 rows, bilingual (EN/ES) vocabulary, no per-language
split in the vectorizer itself (the `clean_comment` tokens are already
language-appropriate stems from `specs/006-text-preprocessing/spec.md`)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Practicality over precision**: TF-IDF/BoW via scikit-learn are the
  standard, lowest-friction baseline for a linear classifier — pass.
- **Reproducibility**: implemented as scriptable functions
  (`src/features/vectorize.py`) plus a script
  (`scripts/build_features.py`), not notebook-only exploration — pass.
- **Incremental delivery**: this is exactly the next atomic task in
  `specs/001-hate-speech-detection/tasks.md` section 3, scoped to
  vectorization only (no model training yet) — pass.
- **Documented code**: functions will carry short docstrings/comments
  matching the style of `src/preprocessing/pipeline.py` — pass.
- No violations requiring the Complexity Tracking table.

## Project Structure

### Documentation (this feature)

```text
specs/049-classic-text-vectorization/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
└── spec.md               # Feature spec (already exists)
```

No `contracts/` directory: this feature has no external API/CLI contract
of its own — it's an internal library step consumed by the next task
(model training) via a plain Python function call, matching how
`src/preprocessing/pipeline.py` and `src/data/spanish_augmentation.py`
are consumed by their respective scripts.

### Source Code (repository root)

```text
src/
└── features/
    ├── __init__.py        # already exists (empty)
    └── vectorize.py        # NEW: build_tfidf_vectorizer, build_bow_vectorizer,
                             #      fit_transform_corpus

scripts/
└── build_features.py       # NEW: loads the augmented preprocessed CSV,
                             #      fits both vectorizers, reports sizes,
                             #      persists the chosen (TF-IDF) vectorizer

tests/
└── test_vectorize.py       # NEW: unit tests against a small fake corpus
```

**Structure Decision**: single-project layout (existing `src/` + `scripts/`
+ `tests/` convention used by every prior feature in this repo — no new
top-level directories needed).

## Complexity Tracking

No Constitution Check violations — table not needed.
