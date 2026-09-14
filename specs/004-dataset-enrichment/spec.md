# Spec: Enrich dataset with public multi-category hate speech datasets

## Linked issue

- GitHub issue: #33
- Kanban status at spec creation time: In Progress
- Level: Nivel Esencial

## What

Combine the briefing dataset (`youtoxic_english_1000.csv`) with three
additional public English hate-speech datasets — HateXplain, ETHOS, and
Measuring Hate Speech (UC Berkeley D-Lab) — into a single enriched dataset,
normalized to a common schema that captures not just a generic hate/not-hate
flag but *which kind* of hate (racism, xenophobia, religion-based hate,
misogyny, homophobia, transphobia, disability-based hate, classism,
violence).

## Why

The briefing dataset alone (1000 rows, 13.8% positive class, single binary
label) is too small and too coarse: a model trained only on it cannot learn
to distinguish *why* a comment is hateful, and is at high risk of
overfitting to that dataset's specific vocabulary and topics (constitution
principle: "Generalization over memorization"). Combining several
independently-collected, differently-annotated datasets gives more data,
more topic diversity, and category-level signal that later tasks (e.g.
ensemble models, richer evaluation) can use.

## Scope

- Download HateXplain (GitHub, `dataset.json`), ETHOS (GitHub, binary +
  multi-label CSVs), and Measuring Hate Speech (HuggingFace, parquet) — all
  freely downloadable without an account.
- Normalize each source to a shared schema:
  `comment, language, label, categories, source`.
- Provide a `combine_datasets()` function that concatenates all normalized
  sources (briefing dataset included) into one DataFrame.
- Document the category-mapping decisions (which raw labels map to which
  category) since they involve judgment calls.

## Out of scope

- HatEval and any other dataset that requires accepting terms on an account
  the project doesn't have — tracked separately in issue #34.
- Actual preprocessing/vectorization/model training on the enriched dataset
  (separate tasks in `specs/001-hate-speech-detection/tasks.md`).
- Perfect category taxonomy alignment across sources — each source has its
  own annotation scheme; mappings are best-effort and documented, not
  guaranteed lossless.

## Success criteria (EARS-style)

- WHEN each source dataset is loaded, THE SYSTEM SHALL produce a DataFrame
  with columns `comment, language, label, categories, source`.
- WHEN HateXplain posts have multiple annotators, THE SYSTEM SHALL resolve
  the final label and target categories by majority vote.
- WHEN a source's hate signal is continuous (e.g. ETHOS's `isHate` score,
  Measuring Hate Speech's `hate_speech_score`), THE SYSTEM SHALL binarize it
  with a documented, fixed threshold.
- WHEN `combine_datasets()` is called with the normalized sources, THE
  SYSTEM SHALL return a single DataFrame with all rows and no column
  mismatches.
- IF a comment's categories cannot be mapped to the project's known category
  vocabulary, THEN THE SYSTEM SHALL tag it `"other"` rather than dropping it
  or raising an error.

## Result

Running `scripts/build_enriched_dataset.py` produces
`data/processed/enriched_comments_en.csv` with **61,711 rows** (up from
1,000), combining:

| Source | Rows |
|---|---|
| `measuring_hate_speech` | 39,565 |
| `hatexplain` | 20,148 |
| `youtoxic` (briefing) | 1,000 |
| `ethos` | 998 |

Label balance improved from 13.8% to **~32.2%** hate (19,874 / 61,711), and
categories now span racism, misogyny, violence, religion, homophobia,
xenophobia, transphobia, disability, and classism (individually and in
combination), instead of a single undifferentiated "hate" flag.

## Open questions

- Should "offensive" (HateXplain's middle label, between "normal" and
  "hatespeech") count as hate for this project's binary target? Resolved:
  no — only the strict "hatespeech" majority label counts, to keep the
  label consistent with the briefing's own "hate speech" framing.
- Spanish coverage remains unresolved until issue #34 (HatEval access) is
  unblocked.
