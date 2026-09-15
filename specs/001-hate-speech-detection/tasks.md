# Tasks: Hate Speech Detection in YouTube Comments

Atomic tasks derived from `plan.md`, written with EARS-style acceptance criteria.

## 1. Data loading

- [x] Load the raw YouTube comments dataset into a DataFrame.
  - WHEN the dataset path is provided, THE SYSTEM SHALL load it into a pandas
    DataFrame with consistent column names.
  - Implemented in `src/data/loader.py::load_comments`, verified by
    `tests/test_data_loader.py` and enforced on every push/PR by the CI
    harness (`.github/workflows/ci.yml`).
  - Real dataset (`data/raw/youtoxic_english_1000.csv`, from the briefing's
    Google Drive link) downloaded and explored via `dataset_summary`:
    1000 rows, 0 nulls, 3 duplicate comments, label imbalance of 862
    not-hate (`IsHatespeech=False`) vs. 138 hate (`IsHatespeech=True`),
    i.e. ~13.8% positive class. The loader maps the dataset's native
    `Text`/`IsHatespeech` columns to the project's `comment`/`label`
    schema via `COLUMN_ALIASES`.
  - The class imbalance (13.8% hate) means accuracy alone will be
    misleading for later evaluation tasks — favor precision/recall/F1 and
    consider class weighting during model training.

## 2. Preprocessing

- [x] Implement text cleaning (lowercasing, URL/mention/emoji removal via regex).
  - WHEN raw comment text is passed in, THE SYSTEM SHALL return cleaned text with
    no URLs, HTML entities, or control characters.
- [x] Implement tokenization and stopword removal.
- [x] Implement stemming/lemmatization.
  - WHEN cleaned text is tokenized, THE SYSTEM SHALL return a list of
    lemmatized/stemmed tokens excluding stopwords.
  - See `specs/006-text-preprocessing/spec.md` for the full design.
    Implemented in `src/preprocessing/{clean,tokenize,stem,pipeline}.py`,
    language-aware (English/Spanish, matching the bilingual enriched
    dataset), verified by `tests/test_preprocessing.py`. Applied to the
    full 133,808-row enriched dataset in 46s via
    `scripts/preprocess_enriched_dataset.py`.

## 3. Feature extraction

- [x] Implement TF-IDF vectorization.
- [x] Implement Bag of Words vectorization for comparison.
  - WHEN a preprocessed corpus is vectorized, THE SYSTEM SHALL produce a fixed-size
    numeric feature matrix usable by scikit-learn estimators.
  - See `specs/049-classic-text-vectorization/spec.md`. Implemented in
    `src/features/vectorize.py`, verified by `tests/test_vectorize.py`.
    Both vectorizers, run via `scripts/build_features.py` against the full
    151,848-row Spanish-augmented dataset
    (`enriched_comments_es_augmented_preprocessed.csv`), share an
    18,417-token vocabulary (min_df=5, 99.94% sparse); TF-IDF was picked
    as the baseline and persisted to
    `data/processed/tfidf_vectorizer.joblib`.

## 4. Model training

- [ ] Train baseline Logistic Regression on TF-IDF features.
- [ ] Train and compare Linear SVM and Multinomial Naive Bayes.
  - WHEN a model is trained, THE SYSTEM SHALL persist it alongside its fitted
    vectorizer for reuse at inference time.

## 5. Evaluation

- [ ] Compute accuracy, precision, recall, F1-score, and confusion matrix on a
      held-out test set.
- [ ] Compare train vs. test metrics for overfitting.
  - IF train/test metric gap exceeds 5 percentage points, THEN THE SYSTEM SHALL
    flag the model as overfit in the evaluation report.

## 6. Hyperparameter tuning

- [ ] Define the hyperparameter search space for the chosen model.
- [ ] Run tuning (Optuna) and record the best parameters and resulting metrics.

## 7. Serving

- [ ] Build a FastAPI endpoint that accepts raw text and returns a hate/not-hate
      prediction.
- [ ] Build a Streamlit app that calls the prediction logic for manual testing.
  - WHEN a user submits a comment through the app, THE SYSTEM SHALL display the
    predicted label within the same session.

## 8. Documentation

- [ ] Document setup, usage, and evaluation results in the project README.
