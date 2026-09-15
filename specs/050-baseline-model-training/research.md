# Phase 0 Research: Baseline model training

## Decision: fit TF-IDF on the training split only, not the spec-049 vectorizer

**Rationale**: `data/processed/tfidf_vectorizer.joblib` (from
`specs/049-classic-text-vectorization/spec.md`) was fit on the *full*
151,848-row corpus, purely to measure feature-space size versus Bag of
Words. Reusing it directly for model training would mean the vectorizer's
IDF weights (and, in principle, its vocabulary pruning via `min_df`) were
computed with knowledge of rows that end up in the test split — a form of
data leakage that undermines the held-out evaluation this task exists to
set up for issue #8. Fitting a second, training-only vectorizer costs
almost nothing (same class, same hyperparameters) and keeps the
train/test separation real.

**Alternatives considered**: reuse the spec-049 vectorizer as-is — simpler
(one less artifact), but rejected given the constitution's explicit
"generalization over memorization" principle; a leaked vectorizer would
quietly inflate test-set scores in exactly the way issue #8's overfitting
check is meant to catch.

## Decision: stratified split, fixed `random_state`

**Rationale**: stratification keeps the ~37.3%/62.7% class balance
consistent across train and test, so neither split accidentally
over/under-represents the minority (hateful) class purely by sampling
luck. A fixed `random_state` (`42`, matching no particular convention in
this repo but consistent within this task) makes the split — and
therefore every downstream metric — reproducible run to run, per the
constitution's "Reproducibility" principle.

**Alternatives considered**: plain random split — rejected, risks a
test-set class balance that drifts from the true ~37.3%, making metrics
harder to compare run-to-run without stratification pinning it down.

## Decision: `LogisticRegression`, `LinearSVC`, `MultinomialNB`

**Rationale**: exactly the three named in issue #7 and
`specs/001-hate-speech-detection/tasks.md` section 4 — all handle sparse
TF-IDF input natively and train in seconds-to-low-minutes on ~18k
features × ~121k rows, matching the "Practicality over precision"
principle. `class_weight="balanced"` is set on `LogisticRegression` and
`LinearSVC` (both support it) to counter the ~63/37 imbalance;
`MultinomialNB` has no `class_weight` parameter (it assumes/estimates
class priors from the data directly), so it's evaluated as-is and its
comparison result documents whether the imbalance hurts it in practice.

**Alternatives considered**: gradient boosting / random forest as a
fourth baseline — rejected as out of scope; issue #7 names three specific
classic linear/NB models, and tree ensembles over ~18k sparse
TF-IDF columns would be both slower and a scope expansion belonging more
naturally to the ensemble task (issue #11).

## Decision: F1 on the hateful class (not accuracy) decides the winner

**Rationale**: `specs/001-hate-speech-detection/tasks.md` section 1
already flags that accuracy alone is misleading under class imbalance.
At ~37.3% hateful, a model predicting "not hate" for everything would
still score ~62.7% accuracy while being useless — F1 on the positive
(hateful) class is reported alongside accuracy, and used as the
tie-breaker for which model gets persisted.

**Alternatives considered**: macro-F1 (average of both classes' F1) —
considered, but the hateful class is the one that actually matters for
this product (flagging hate speech), so per-class F1 on `label == 1` is
more directly aligned with the project's goal than an average that
partly rewards not-hate performance.
