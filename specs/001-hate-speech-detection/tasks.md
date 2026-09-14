# Tasks: Hate Speech Detection in YouTube Comments

Atomic tasks derived from `plan.md`, written with EARS-style acceptance criteria.

## 1. Data loading

- [ ] Load the raw YouTube comments dataset into a DataFrame.
  - WHEN the dataset path is provided, THE SYSTEM SHALL load it into a pandas
    DataFrame with consistent column names.

## 2. Preprocessing

- [ ] Implement text cleaning (lowercasing, URL/mention/emoji removal via regex).
  - WHEN raw comment text is passed in, THE SYSTEM SHALL return cleaned text with
    no URLs, HTML entities, or control characters.
- [ ] Implement tokenization and stopword removal.
- [ ] Implement stemming/lemmatization.
  - WHEN cleaned text is tokenized, THE SYSTEM SHALL return a list of
    lemmatized/stemmed tokens excluding stopwords.

## 3. Feature extraction

- [ ] Implement TF-IDF vectorization.
- [ ] Implement Bag of Words vectorization for comparison.
  - WHEN a preprocessed corpus is vectorized, THE SYSTEM SHALL produce a fixed-size
    numeric feature matrix usable by scikit-learn estimators.

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
