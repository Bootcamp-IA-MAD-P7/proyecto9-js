# Spec: Auxiliary sarcasm dataset (English + machine-translated Spanish)

## Linked issue

- GitHub issue: #39
- Kanban status at spec creation time: In Progress
- Level: Nivel Esencial

## What

Add a sarcasm/irony detection dataset covering English (native) and Spanish
(machine-translated), kept as a **separate auxiliary dataset** from the
hate-speech dataset in `specs/004-dataset-enrichment/spec.md` — not merged
into its `categories` column.

## Why

Sarcastic and ironic hate speech is a known hard case for classifiers:
sarcasm can flip the surface sentiment of a hateful comment, or make a
non-hateful comment look aggressive on the surface. Having a sarcasm signal
available (even from a separate corpus) lets later modeling work explore it
as an auxiliary feature (e.g. a sarcasm-probability score as extra input to
the hate classifier) or a specific error-analysis axis.

## Why a separate dataset instead of a new category

The hate-speech dataset's `categories` column (racism, misogyny, etc.)
describes *what kind of hate* a **hateful** comment expresses. Sarcasm is a
different, orthogonal dimension — a comment can be sarcastic and hateful,
sarcastic and harmless, or neither. The only public, freely-downloadable
sarcasm dataset with actual text (not just Twitter IDs requiring a paid API
to hydrate) is a **news headlines** corpus (TheOnion vs. HuffPost) — a
completely different domain from YouTube/Twitter comments, with no row-level
overlap with the hate dataset. Forcing it into the same table would create
rows with a hate `label` that doesn't apply (headlines were never annotated
for hate speech). Keeping it separate is the honest representation of what
the data actually supports.

## Scope

- Load the English source (`sarcasm_headlines_en.json`, ~28,600 headlines,
  binary `is_sarcastic` label) into a shared schema:
  `comment, language, is_sarcastic, source`.
- Generate a Spanish version by machine-translating the English headlines
  with a local model (`Helsinki-NLP/opus-mt-en-es` via `transformers`), not
  a rate-limited free translation API (tested: MyMemory's anonymous tier
  caps at 5,000 characters/day, unusable at this scale).
- Document that the Spanish version is **synthetic/translated**, not
  native Spanish sarcasm, and that both English and Spanish sarcasm data
  come from a different domain (news headlines) than the hate dataset.

## Out of scope

- Merging sarcasm labels into the hate-speech dataset's schema or rows.
- Native Spanish sarcasm/irony corpora: investigated IroSvA (IberLEF) but
  it only distributes Twitter **IDs**, not text — hydrating them requires
  paid X/Twitter API access, which this project doesn't have.
- Using sarcasm as an actual model input feature — that's a future
  modeling task, not part of this data-preparation spec.

## Success criteria (EARS-style)

- WHEN the English sarcasm source is loaded, THE SYSTEM SHALL produce a
  DataFrame with columns `comment, language, is_sarcastic, source`.
- WHEN the English DataFrame is translated, THE SYSTEM SHALL produce a
  Spanish DataFrame with the same `is_sarcastic` labels and row count,
  tagged with a `source` suffix (`_es_mt`) marking it as machine-translated.
- WHERE the real translation model is unavailable (heavy dependency not
  installed), THE SYSTEM SHALL still allow `build_spanish_sarcasm()` to be
  tested via dependency injection (`translate_fn`), without requiring
  `torch`/`transformers` in the default test/CI environment.

## Result

`scripts/build_sarcasm_dataset.py` produces
`data/processed/sarcasm_comments.csv`: ~28,619 English headlines +
~28,619 machine-translated Spanish headlines ≈ 57,238 rows, roughly balanced
between sarcastic/non-sarcastic (the source dataset's own split).
Translation runs locally via MarianMT, taking ~80-90 minutes on CPU for the
full corpus (measured: ~0.18s/headline). No per-day rate limit, unlike free
translation APIs.

## Open questions

- Translation quality is not manually validated at scale; a small manual
  spot-check is recommended before using the Spanish sarcasm data for
  anything beyond experimentation.
- Whether sarcasm should later become an actual input feature for the hate
  classifier (e.g. via a separately-trained sarcasm-detection model applied
  to the hate dataset's own comments) is left for a future spec.
