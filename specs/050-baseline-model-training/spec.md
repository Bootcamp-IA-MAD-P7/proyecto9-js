# Spec 050: Baseline hate speech classification model

## Linked issue

- GitHub issue: #7
- Kanban status at spec creation time: In Progress
- Level: Nivel Esencial

## What

Train three classic scikit-learn classifiers — Logistic Regression,
Linear SVM, and Multinomial Naive Bayes — on the TF-IDF-vectorized,
Spanish-augmented corpus, compare them on a held-out split, and persist
the best-performing model alongside a matched, freshly-fit TF-IDF
vectorizer for reuse at inference time.

## Why

`specs/001-hate-speech-detection/tasks.md` section 4 ("Model training")
is the next unblocked task now that feature extraction
(`specs/049-classic-text-vectorization/spec.md`) is done. A full
evaluation harness (metrics, overfitting checks) is a separate, later
task (issue #8, `specs/001-hate-speech-detection/tasks.md` section 5),
but picking *which* of the three classic algorithms to carry forward
needs at least a basic train/test comparison now, otherwise issue #8
would have nothing concrete to evaluate.

## Scope

- `src/models/train.py`:
  - `split_dataset()`: stratified train/test split on `label` (holds the
    class balance — 37.3% hateful per
    `specs/048-spanish-category-augmentation/spec.md` — roughly constant
    across both splits).
  - `build_models()`: returns the three untrained estimators
    (`LogisticRegression`, `LinearSVC`, `MultinomialNB`), with
    `class_weight="balanced"` on the two that support it, to avoid the
    majority (not-hate) class dominating the fit given the ~63/37
    imbalance.
  - `train_and_compare()`: fits each model on the TF-IDF-transformed
    training split, scores all three on the held-out test split
    (accuracy and F1 on the hateful class — accuracy alone is misleading
    at this imbalance, per `specs/001-hate-speech-detection/tasks.md`
    section 1's own finding), and returns per-model metrics plus the
    winning model.
- `scripts/train_baseline_model.py`: loads
  `data/processed/enriched_comments_es_augmented_preprocessed.csv`,
  splits it, **fits a fresh TF-IDF vectorizer on the training split only**
  (not the one persisted by spec 049, which was fit on the full corpus
  purely to compare TF-IDF vs. BoW feature-space size — reusing it here
  would leak test-set vocabulary/IDF statistics into training), runs
  `train_and_compare()`, prints the comparison table, and persists the
  winning model + its matched vectorizer.
- `tests/test_train.py`: unit tests against a small fake corpus.

## EARS criteria

1. WHEN the preprocessed corpus is loaded, THE SYSTEM SHALL split it into
   training and test sets using stratified sampling on `label`, so both
   splits preserve the ~37.3% hateful class balance.
2. WHEN the TF-IDF vectorizer is fit, THE SYSTEM SHALL fit it on the
   training split only, and use that same fitted vectorizer (never
   refit) to transform the test split, so no test-set information leaks
   into the feature space.
3. WHEN each of the three models is trained, THE SYSTEM SHALL evaluate it
   on the untouched test split and report both accuracy and F1-score on
   the hateful class, so the comparison isn't distorted by class
   imbalance.
4. THE SYSTEM SHALL persist the best-performing model (by F1 on the
   hateful class) together with its exact training-fit vectorizer as a
   matched pair, so inference-time transform/predict use consistent
   feature indices.
5. THE SYSTEM SHALL expose `split_dataset`, `build_models`, and
   `train_and_compare` as independently testable functions accepting
   plain DataFrames/arrays, so tests run against a small fake corpus
   without needing the full 151,848-row dataset.

## Result

Ran `scripts/train_baseline_model.py` against the full augmented,
preprocessed dataset (151,848 rows), stratified 80/20 split (train:
121,478 rows, test: 30,370 rows, both 37.3% hateful, matching
`specs/048-spanish-category-augmentation/spec.md`'s class balance).
TF-IDF was fit on the training split only (18,417-token vocabulary at
`min_df=5`, same threshold as spec 049 but a training-only-fit
vocabulary, not the full-corpus one).

| Model | Accuracy | F1 (hateful) |
|---|---|---|
| **Logistic Regression** | **79.14%** | **0.7267** (winner) |
| Linear SVM | 78.35% | 0.7188 |
| Multinomial Naive Bayes | 78.06% | 0.6546 |

Logistic Regression won on F1 for the hateful class (the tie-breaker
metric per this spec, since accuracy alone is misleading at this class
balance) and was persisted to `data/processed/baseline_model.joblib`
(129 KB) alongside its matched, training-only-fit vectorizer at
`data/processed/baseline_vectorizer.joblib` (318 KB). Naive Bayes
trailed the other two by a wider F1 margin than accuracy alone suggests
(6.5-7.2 points of F1 vs. ~1 point of accuracy), consistent with it
having no `class_weight` lever to counter the ~63/37 imbalance —
something issue #8's fuller evaluation should look at more closely
(e.g. a confusion matrix breakdown) rather than something this task
needed to fix.

## Out of scope

- Full evaluation report (confusion matrix, train/test overfitting-gap
  check, per-category breakdown) — issue #8,
  `specs/001-hate-speech-detection/tasks.md` section 5.
- Hyperparameter tuning — issue #14 (Optuna), later/advanced level.
- Ensemble model combining multiple classifiers — issue #11, later/
  advanced level (see the two-track plan discussed with the user: simple
  baselines first, ensemble compared against them afterward, not a
  replacement decided in advance).
- Serving the model via API/UI — issue #9,
  `specs/001-hate-speech-detection/tasks.md` section 7.
