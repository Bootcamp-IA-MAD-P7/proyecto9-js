# Spec 049: Classic text vectorization (TF-IDF / Bag of Words)

## Linked issue

- GitHub issue: #6
- Kanban status at spec creation time: To Do
- Level: Nivel Esencial

## What

Vectorize the preprocessed, bilingual (English + Spanish) `clean_comment`
corpus using two classic representations — TF-IDF and Bag of Words —
compare their resulting feature-space sizes, and pick a baseline
representation for the model-training task
(`specs/001-hate-speech-detection/tasks.md` section 4, issue #7).

## Why

Section 3 of `specs/001-hate-speech-detection/tasks.md` ("Feature
extraction") is the next unblocked task now that preprocessing
(`specs/006-text-preprocessing/spec.md`) and the Spanish augmentation
follow-up (`specs/048-spanish-category-augmentation/spec.md`) are both
done. Per the project decision after spec 048, this and all future work
use the **augmented** dataset
(`data/processed/enriched_comments_es_augmented_preprocessed.csv`,
151,848 rows) rather than the pre-augmentation one, so the 7 previously
zero-coverage categories are represented in Spanish too.

TF-IDF and Bag of Words are both needed as a comparison point: BoW is the
simpler baseline (raw token counts), TF-IDF down-weights tokens that are
common across the whole corpus (e.g. "url", generic stopword-adjacent
stems that survived preprocessing) in favor of tokens that are
distinctive of a given comment. Comparing their feature-space sizes and a
quick baseline-model sanity check informs which one downstream model
training (issue #7) should default to.

## Scope

- `src/features/vectorize.py`:
  - `build_tfidf_vectorizer()` / `build_bow_vectorizer()`: thin,
    configurable wrappers around scikit-learn's `TfidfVectorizer` /
    `CountVectorizer`, sharing the same token pattern and `min_df` so the
    two are comparable apples-to-apples.
  - `fit_transform_corpus()`: fits a given vectorizer on the
    `clean_comment` column and returns the sparse feature matrix plus the
    fitted vectorizer (for reuse at inference time, per
    `specs/001-hate-speech-detection/tasks.md` section 4).
- `scripts/build_features.py`: loads
  `data/processed/enriched_comments_es_augmented_preprocessed.csv`, fits
  both vectorizers, reports feature-space size (vocabulary size, matrix
  shape, sparsity) for each, and records which one is picked as the
  baseline.
- `tests/test_vectorize.py`: unit tests against a small fake corpus (no
  need to load the full 151,848-row dataset in tests).

## EARS criteria

1. WHEN the preprocessed corpus is loaded, THE SYSTEM SHALL vectorize the
   `clean_comment` column using both TF-IDF and Bag of Words, sharing the
   same tokenization/vocabulary-filtering configuration so the two
   feature spaces are directly comparable.
2. WHEN a vectorizer is fit, THE SYSTEM SHALL expose both the resulting
   sparse feature matrix and the fitted vectorizer object, so the
   vectorizer can be persisted and reused unchanged at inference time.
3. THE SYSTEM SHALL report, for each representation, the vocabulary size
   and matrix sparsity, so the comparison in the Result section below is
   grounded in actual numbers rather than assumptions.
4. THE SYSTEM SHALL NOT modify `clean_comment` or any other existing
   column — vectorization is a read-only, additive step over the
   preprocessed dataset.

## Result

Ran `scripts/build_features.py` against the full augmented, preprocessed
dataset (151,848 rows). With a shared `min_df=5` vocabulary-pruning
threshold, both vectorizers land on the **same 18,417-token vocabulary**
(same tokens, only the cell values differ) and the same matrix shape
`(151848, 18417)`, at **99.94% sparsity** (1,717,690 non-zero entries out
of ~2.8 billion cells).

Since TF-IDF and BoW share the exact vocabulary here (they always will,
given identical `token_pattern`/`min_df`), the deciding factor for the
baseline isn't feature-space size but the value semantics: TF-IDF was
picked because it down-weights tokens that are common across the whole
151,848-row corpus (e.g. generic stems surviving preprocessing) in favor
of tokens distinctive of a given comment, without needing a second,
corpus-specific stopword pass. The fitted TF-IDF vectorizer is persisted
at `data/processed/tfidf_vectorizer.joblib` (367 KB) for reuse, unchanged,
at inference time by the model-training task (issue #7).

## Out of scope

- Word embeddings / transformer-based representations — separate,
  later task (`specs/001-hate-speech-detection/tasks.md` section 3 only
  covers classic vectorization; embeddings are issue #19).
- Model training and evaluation — issue #7/#8, builds on top of whichever
  representation is picked here.
- Re-vectorizing the pre-augmentation dataset
  (`enriched_comments_preprocessed.csv`) — superseded by the augmented
  dataset per the project decision after spec 048.
