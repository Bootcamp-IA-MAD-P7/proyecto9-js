# Spec: Formal EDA on the enriched, preprocessed dataset

## Linked issue

- GitHub issue: #46
- Kanban status at spec creation time: In Progress
- Level: Nivel Esencial

## What

A formal exploratory data analysis (EDA) of the combined 133,808-row
enriched dataset (`specs/004-dataset-enrichment/spec.md`), after applying
the text preprocessing pipeline (`specs/006-text-preprocessing/spec.md`),
delivered as a Jupyter notebook.

## Why

The dataset grew from the briefing's original 1,000 rows to 133,808 across
9 sources, 2 languages, and 9 category tags — far too large and complex to
understand from summary statistics printed in a terminal. A proper EDA
with visualizations is needed before committing to a vectorization/model
approach, both to catch data-quality issues early and to inform modeling
decisions (class weighting, per-language handling, category coverage
gaps).

## Scope

- `notebooks/eda_enriched_dataset.ipynb` covering: dataset overview,
  missing values, class balance, language balance, source distribution,
  category distribution, comment length, duplicates, top tokens (post
  preprocessing), word clouds, and a category × language cross-tab.
- Every finding is written as a markdown conclusion directly under its
  plot, grounded in the actual computed numbers (not assumptions written
  before running the cells — see the Result section below for a
  self-correction that happened during this work).
- **The notebook must be committed already executed**, with every plot
  embedded as a static output, so opening it on GitHub (or any notebook
  viewer) shows the finished analysis without re-running anything.

## Out of scope

- Feature engineering or vectorization — that's the next task in
  `specs/001-hate-speech-detection/tasks.md` (section 3).
- Automated data-quality fixes based on the findings (e.g. deduplication)
  — this spec only analyzes and documents, it doesn't modify the dataset.

## Success criteria (EARS-style)

- WHEN the notebook is opened on GitHub without running it, THE SYSTEM
  SHALL display every plot and every finding as already-rendered content.
- WHEN a markdown finding states a number (percentage, count), THE SYSTEM
  SHALL match the actual output of the code cell(s) it follows.
- WHERE a finding text was written before the corresponding cell was
  executed, THE SYSTEM SHALL be corrected against the real output before
  the notebook is considered complete (see Result: an incorrect "55% of
  hateful comments have no category" claim was caught and fixed to the
  real 18.4%).

## Result

Key findings from the executed notebook:

- **Class balance:** 28.8% hate (38,562 / 133,808) — moderate imbalance,
  not severe.
- **Language balance:** 55.8% English / 44.2% Spanish — near parity at
  the row level.
- **Category balance is uneven despite row-level language parity:**
  `violence` (10,160) and `xenophobia` (10,059) are the top categories;
  **`classism` has exactly 1 tagged comment** in the whole dataset, and
  `disability` (681) / `transphobia` (432) are also thin. Only 18.4% of
  hateful comments carry no category at all (initial draft of this
  finding wrongly assumed ~55% before checking the real output — caught
  and corrected).
- **Comment length is a weak signal:** hate comments trend shorter (mean
  128 vs. 149 characters) but both distributions strongly overlap.
- **Duplicates are negligible:** 368 exact duplicates (0.28%), and 339 of
  343 duplicate groups occur within a single source (not cross-source
  contamination).
- **No missing values** in any column.
- Top-tokens and word-cloud analysis confirm the `clean_comment` column
  (from the preprocessing pipeline) carries clear separating signal
  between hate/not-hate in both languages, validating the pipeline before
  moving to vectorization.

## Open questions

- None outstanding for this spec. Follow-up sourcing work (more Spanish
  racism/violence/misogyny data, any classism-specific data at all) is
  tracked as a recommendation in the notebook's conclusions, not a new
  task here.
