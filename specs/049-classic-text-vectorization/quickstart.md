# Quickstart: Classic text vectorization

## Prerequisites

- `data/processed/enriched_comments_es_augmented_preprocessed.csv` exists
  (see `specs/048-spanish-category-augmentation/spec.md` and
  `scripts/preprocess_enriched_dataset.py --input
  data/processed/enriched_comments_es_augmented.csv --output
  data/processed/enriched_comments_es_augmented_preprocessed.csv`).
- Base `requirements.txt` installed (scikit-learn is already a core
  dependency, no `requirements-nlp.txt` needed for this step — no
  translation model involved).

## Run

```bash
python scripts/build_features.py
```

## Expected outcome

- Prints, for both `tfidf` and `bow`: vocabulary size, non-zero entry
  count, and sparsity.
- Writes `data/processed/tfidf_vectorizer.joblib` (the chosen baseline).
- Exit code 0, no exceptions.

## Validate

```bash
python -m pytest tests/test_vectorize.py -q
```

All tests should pass, covering:
- both vectorizers produce a sparse matrix with the expected row count
- shared configuration means BoW's vocabulary is a superset of/equal to
  TF-IDF's for the same `min_df` (same tokenizer, same pruning threshold)
- the fitted vectorizer's `.transform()` on new text reuses the same
  vocabulary (no refitting)
