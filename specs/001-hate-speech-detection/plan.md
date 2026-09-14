# Plan: Hate Speech Detection in YouTube Comments

## Architecture overview

```
src/
├── data/            # Data loading and raw dataset handling
├── preprocessing/    # Text cleaning, tokenization, stopwords, stemming/lemmatization
├── features/         # Vectorization (BoW, TF-IDF)
├── models/           # Training, hyperparameter tuning, persisted model artifacts
├── evaluation/        # Metrics, confusion matrix, overfitting checks
├── api/               # Prediction service (FastAPI)
└── app/                # User-facing interface (Streamlit)
```

## Technical decisions

- **Language:** Python 3.11+.
- **Core libraries:** pandas, scikit-learn, NLTK/spaCy for NLP preprocessing,
  Optuna for hyperparameter tuning.
- **Vectorization:** start with TF-IDF (baseline), compare against Bag of Words.
- **Models to compare:** Logistic Regression, Linear SVM, Multinomial Naive Bayes.
  Pick the best by F1-score on validation, then tune it.
- **Interface:** Streamlit app for manual queries; a thin FastAPI service backing
  it so the same prediction logic is reusable from other clients.
- **Environment:** `requirements.txt` for dependency pinning; Docker support
  planned for a later delivery level.

## Data flow

1. Raw dataset (`data/raw/`) is loaded via `src/data`.
2. `src/preprocessing` cleans text: lowercasing, URL/mention/emoji handling via
   regex, stopword removal, stemming/lemmatization.
3. `src/features` vectorizes the cleaned corpus.
4. `src/models` trains and tunes a classifier on the vectorized features.
5. `src/evaluation` computes metrics and checks the overfitting threshold.
6. The trained model + vectorizer are persisted (`models/` artifacts) and loaded
   by `src/api` for inference, consumed by `src/app`.

## Milestones

1. Dataset exploration and preprocessing pipeline.
2. Baseline model (TF-IDF + Logistic Regression) with evaluation report.
3. Model comparison and hyperparameter tuning.
4. Prediction API + Streamlit interface wired to the trained model.
5. Documentation and README covering setup, usage, and results.

## Risks

- Class imbalance in hate/not-hate labels may skew accuracy; mitigate with
  precision/recall/F1 reporting and class weighting, not accuracy alone.
- Overfitting risk with high-dimensional TF-IDF features on a small dataset;
  mitigate with regularization and cross-validation during tuning.
