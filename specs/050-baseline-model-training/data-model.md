# Phase 1 Data Model: Baseline model training

No persistent database entities — this feature reads the existing corpus
and writes fitted-object artifacts.

## Preprocessed corpus (input)

Already defined by `specs/048-spanish-category-augmentation/spec.md` /
`specs/006-text-preprocessing/spec.md`; read-only here.

- Source: `data/processed/enriched_comments_es_augmented_preprocessed.csv`
- Relevant columns: `clean_comment` (str, features), `label` (int, 0/1
  target)
- Row count: 151,848 (~37.3% `label == 1`)

## Train/test split (intermediate, in-memory only)

- `X_train`, `X_test`: `clean_comment` values (str arrays)
- `y_train`, `y_test`: `label` values (int arrays)
- Split ratio: 80/20, `stratify=y`, `random_state=42`
- Not persisted as files — regenerated deterministically on each run via
  the fixed `random_state`

## Fitted vectorizer (output)

- Type: `sklearn.feature_extraction.text.TfidfVectorizer`, fit on
  `X_train` only (per research.md's leakage-avoidance decision)
- Persisted as: `data/processed/baseline_vectorizer.joblib`

## Trained models (intermediate + one output)

- Three fitted estimators: `LogisticRegression`, `LinearSVC`,
  `MultinomialNB`, each fit on the TF-IDF-transformed `X_train`/`y_train`
- Only the winner (by hateful-class F1 on `X_test`) is persisted, as
  `data/processed/baseline_model.joblib`

## Comparison report (script output, not persisted as a file)

- One row per model: `accuracy`, `f1_hateful` (F1 on `label == 1`)
- Printed by `scripts/train_baseline_model.py`; the concrete numbers get
  copied into spec.md's Result section once the script has run, matching
  how specs 048/049 recorded their real run results.
