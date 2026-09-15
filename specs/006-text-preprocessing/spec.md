# Spec: Text preprocessing pipeline

## Linked issue

- GitHub issue: #5
- Kanban status at spec creation time: Todo
- Level: Nivel Esencial

## What

A language-aware text preprocessing pipeline — cleaning, tokenization,
stopword removal, and stemming — applied to the enriched, bilingual
(English + Spanish) hate-speech dataset from
`specs/004-dataset-enrichment/spec.md`.

## Why

Classic NLP vectorization (TF-IDF / Bag of Words, the next task) needs
normalized text: raw comments are full of URLs, @mentions, emoji, HTML
entities, and language-specific stopwords that add noise without signal.
Because the dataset is ~44% Spanish and ~56% English (see
`specs/004-dataset-enrichment/spec.md`'s Result section), the pipeline must
pick the right stopword list and stemmer **per row**, not assume a single
language.

## Scope

- `clean_text()`: lowercase, strip URLs, @mentions, HTML entities, control
  characters, and emoji via regex.
- `tokenize()` + `remove_stopwords()`: split into word tokens and drop
  English/Spanish stopwords based on the row's `language` column.
- `stem_tokens()`: reduce tokens to a common root using NLTK's Snowball
  stemmer (English and Spanish) — chosen over lemmatization because it
  needs no downloaded corpora beyond stopwords (the stemmer itself is a
  pure algorithm), keeping the harness fast and dependency-light.
- `preprocess()` / `preprocess_dataframe()`: the combined pipeline,
  producing a `clean_comment` column.

## Out of scope

- Vectorization (TF-IDF/BoW) — next task in
  `specs/001-hate-speech-detection/tasks.md`.
- Data augmentation (translation, synonym replacement) — a separate
  evaluation-criteria task from the briefing, not part of this spec.
- Fixing any upstream data-quality issues in the source datasets
  themselves (this pipeline only transforms text, it doesn't correct
  mislabeled rows).

## Success criteria (EARS-style)

- WHEN raw comment text is passed to `clean_text()`, THE SYSTEM SHALL
  return text with no URLs, @mentions, HTML entities, control characters,
  or emoji, lowercased.
- WHEN cleaned text is tokenized, THE SYSTEM SHALL return a list of
  stemmed tokens with stopwords excluded, using the stopword list and
  stemmer matching the comment's `language` ("en" or "es").
- WHEN `preprocess_dataframe()` is called on the enriched dataset, THE
  SYSTEM SHALL add a `clean_comment` column without altering row count or
  other columns.

## Result

Ran `scripts/preprocess_enriched_dataset.py` against the full enriched
dataset (133,808 rows): completed in **46 seconds**, producing
`data/processed/enriched_comments_preprocessed.csv` with a new
`clean_comment` column. No encoding issues found in the source data itself
(a `�` seen in ad hoc terminal output during development was a console
display artifact, not a real character in the underlying UTF-8 files —
verified against the raw bytes).

## Open questions

- None outstanding; lemmatization (a heavier alternative producing
  dictionary-valid word forms instead of stems) was considered but
  deferred — Snowball stemming already satisfies the EARS criterion
  without adding spaCy language models as a dependency for this task.
