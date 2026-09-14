# Spec: Hate Speech Detection in YouTube Comments

## What

A machine learning system that classifies YouTube comments as **hate speech** or
**not hate speech**, plus a user-facing way to query it (interface, API, or both).

## Why

YouTube's moderation team cannot keep up with the volume of hateful comments.
Scaling the human team is prohibitively expensive and doesn't scale with platform
growth. An automated first-pass classifier lets moderators (or automated actions)
focus on flagged content instead of reading everything.

## Scope (this spec)

- Analyze the provided YouTube comments dataset.
- Preprocess the raw text (cleaning, tokenization, stopwords, stemming/lemmatization,
  regular expressions).
- Apply classic NLP vectorization techniques (e.g. Bag of Words, TF-IDF).
- Train a classic ML classification model on the vectorized text.
- Evaluate the model (precision, recall, F1, confusion matrix) and keep the
  train/test metric gap under 5 percentage points.
- Tune hyperparameters for the chosen model.
- Expose the trained model through a simple interface or API so a user can submit
  a message and get a hate/not-hate prediction.

## Out of scope (this spec)

- Ensemble models, video-link based comment scraping, real-time tracking,
  neural networks/transformers, public deployment, Docker, database storage,
  and experiment tracking — these belong to later specs (medium/advanced/expert
  delivery levels).

## Success criteria (EARS-style)

- WHEN a raw comments dataset is provided, THE SYSTEM SHALL produce a cleaned,
  preprocessed dataset ready for vectorization.
- WHEN the preprocessed dataset is vectorized, THE SYSTEM SHALL produce numeric
  feature representations suitable for training a classifier.
- WHEN the model is trained, THE SYSTEM SHALL report accuracy, precision, recall,
  and F1-score on a held-out test set.
- IF the difference between training and test performance exceeds 5 percentage
  points, THEN THE SYSTEM SHALL be considered overfit and require retuning.
- WHEN a user submits a text message through the interface/API, THE SYSTEM SHALL
  return a hate / not-hate classification.
- WHERE hyperparameter tuning is applied, THE SYSTEM SHALL document the search
  space and the final chosen parameters.

## Open questions

- Which specific classic ML algorithm(s) will be compared first (Logistic
  Regression, SVM, Naive Bayes)? To be decided in `plan.md`.
- Interface choice: Streamlit app vs. plain API (FastAPI/Flask)? To be decided in
  `plan.md`.
