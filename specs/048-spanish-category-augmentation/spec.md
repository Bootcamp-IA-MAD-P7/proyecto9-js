# Spec 048: Spanish augmentation for zero-coverage categories

## Context

The EDA in specs/007-eda-enriched-dataset/spec.md found that 7 of the 10
categories in the enriched dataset have **zero or near-zero Spanish rows**:
classism (1 total, 0 ES), disability (681 EN / 0 ES), homophobia (3,315 EN /
0 ES), racism (8,183 EN / 0 ES), religion (4,290 EN / 0 ES), transphobia
(432 EN / 0 ES), violence (10,160 EN / 0 ES).

Public Spanish datasets researched for these categories were found to be
insufficient on their own:
- **HOMO-MEX** (18,200 Mexican-Spanish tweets, homophobia/transphobia):
  requires a registration form via Codabench
  (https://www.codabench.org/competitions/2229/, linked from
  https://portal.odesia.uned.es/en/dataset/homo-mex-2024) — a reasonable
  option to pursue in parallel, but not an immediate unblock.
- **AMI (IberEval 2018, misogyny)**: no direct download link found; would
  require emailing the task organizers.
- **VIRC** (`oeg/virc` on HuggingFace): public, but only 3,256 Spanish rows,
  span-annotated news headlines (domain mismatch with YouTube comments).
- **"Hate Speech Library in Spanish"** (esaidh266 GitHub repo): an Excel
  lexicon of ~7,210 hateful lemmas, not a labeled comment corpus — usable
  only for keyword weak-labeling, not as training data directly.

## Decision

Reuse the local MarianMT translation pipeline already built for the
sarcasm dataset (`src/data/translate.py`,
`Helsinki-NLP/opus-mt-en-es`, no rate limits) to translate the existing
English hateful comments in the 7 orphan categories into Spanish. A manual
spot-check translating real slur-bearing English comments from our own
dataset confirmed the model produces natural Spanish output, including the
exact slur ("maricón") the category is meant to capture:

```
EN: stop being such a fag and grow up
ES: Deja de ser maricón y madura.
```

This is synthetic (translated, not native) Spanish, so translated rows are
tagged with a `_es_mt` source suffix (e.g. `ethos_es_mt`) to keep them
distinguishable from native Spanish rows in any future analysis or
modeling split.

## EARS criteria

1. WHEN the enriched dataset is loaded, THE SYSTEM SHALL select every
   English (`language == "en"`), hateful (`label == 1`) row whose
   `categories` field contains at least one of: homophobia, racism,
   violence, religion, disability, transphobia, classism.
2. WHEN a candidate row is translated, THE SYSTEM SHALL produce a new row
   with `language = "es"`, the same `label` and `categories` as the
   source row, and `source` set to the original source plus the `_es_mt`
   suffix.
3. THE SYSTEM SHALL NOT modify or duplicate rows outside the selected
   categories.
4. THE SYSTEM SHALL expose the selection and translation steps as
   independently testable functions (`select_translation_candidates`,
   `build_spanish_translations`) accepting an injectable translation
   function, so tests do not require `torch`/`transformers`.

## Implementation

- `src/data/spanish_augmentation.py`: `ORPHAN_CATEGORIES`,
  `select_translation_candidates()`, `build_spanish_translations()`.
- `scripts/build_spanish_augmentation.py`: loads
  `data/processed/enriched_comments.csv`, translates the orphan-category
  rows, writes `data/processed/enriched_comments_es_augmented.csv`
  (original rows + translated rows).
- `tests/test_spanish_augmentation.py`: 4 tests covering selection
  filtering and translation-row shape, using a fake translate function.

## Result

<!-- Filled in after the full translation run. -->

## Out of scope

- Pursuing HOMO-MEX registration (recommended to the user as a parallel,
  native-data track, not blocking this synthetic-augmentation work).
- Building a custom scraped dataset (a heavier, separate initiative).
- Wiring the augmented file into the preprocessing pipeline / EDA notebook
  automatically — those were built against `enriched_comments.csv` and can
  be re-pointed at `enriched_comments_es_augmented.csv` in a follow-up task
  once the augmented data is reviewed.
