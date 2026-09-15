# Phase 1 Data Model: Classic text vectorization

This feature has no persistent database entities — it transforms an
existing in-memory/CSV corpus into feature matrices. The "entities" below
are the shapes passed between functions.

## `clean_comment` corpus (input)

Already defined by `specs/006-text-preprocessing/spec.md`; this feature
only reads it, never modifies it (spec.md EARS criterion 4).

- Source: `data/processed/enriched_comments_es_augmented_preprocessed.csv`
- Relevant column: `clean_comment` (str) — whitespace-joined, stemmed,
  stopword-free tokens, already language-appropriate per row
- Row count: 151,848

## Fitted vectorizer (output)

- Type: `sklearn.feature_extraction.text.TfidfVectorizer` or
  `CountVectorizer` (same shape/interface for either)
- Key attributes used downstream:
  - `vocabulary_`: token → column-index mapping (defines feature-space
    size)
  - `.transform(texts)`: reused unchanged at inference time (issue #7),
    so new comments are projected into the same feature space the model
    was trained on
- Persisted as: `data/processed/tfidf_vectorizer.joblib` (the chosen
  baseline; the BoW vectorizer is fit for comparison only, not persisted,
  since spec.md's Scope names TF-IDF/BoW as a comparison to *pick* a
  baseline, not to ship both)

## Feature matrix (output)

- Type: `scipy.sparse.csr_matrix`, shape `(151848, vocab_size)`
- One row per corpus row (same order as the input DataFrame, so it can be
  zipped back with `label`/`language`/`categories` for model training)
- Reported metrics (spec.md EARS criterion 3): `vocab_size` (`.shape[1]`),
  `nnz` (non-zero entries), `sparsity = 1 - nnz / (rows * vocab_size)`

## Comparison report (script output, not persisted as a file)

- One row per representation (`tfidf`, `bow`), columns: `vocab_size`,
  `nnz`, `sparsity`
- Printed by `scripts/build_features.py`, and the concrete numbers get
  copied into spec.md's Result section once the script has run — matching
  how `specs/048-spanish-category-augmentation/spec.md`'s Result section
  was filled in after its build script ran.
