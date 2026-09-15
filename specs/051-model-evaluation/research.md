# Phase 0 Research: Model evaluation and overfitting control

## Decision: reconstruct the split via `split_dataset`, don't persist indices

**Rationale**: `src/models/train.py::split_dataset` is already
deterministic (`random_state=42`, stratified). Calling it again with the
same arguments on the same source CSV reproduces byte-identical
train/test membership without needing to persist row indices as a new
artifact. This keeps the evaluation task dependency-free of anything
beyond what spec 050 already produced.

**Alternatives considered**: have `scripts/train_baseline_model.py`
additionally persist the train/test row indices — rejected as
unnecessary extra state; determinism from a fixed `random_state` already
solves the same problem with zero added artifacts, and is the same
assumption spec 050 already relies on for its own reproducibility.

## Decision: metrics on the hateful class only, plus confusion matrix

**Rationale**: matches spec 050's `f1_hateful` convention (issue #7's
tie-break metric was F1 on `label == 1`, since that's the class the
product cares about). Issue #8 asks for accuracy, precision, recall, F1,
and confusion matrix explicitly — the confusion matrix is what lets a
reader see *why* precision/recall differ (e.g. false positives vs. false
negatives) without needing a separate breakdown.

**Alternatives considered**: macro-averaged precision/recall/F1 (average
across both classes) — rejected for the same reason spec 050 rejected it:
the hateful class is what the product needs to get right, and macro
averaging would partly reward not-hate performance instead.

## Decision: 5-point threshold on *any* of accuracy/precision/recall/F1

**Rationale**: the constitution states "the gap between training and
test metrics must stay under 5 percentage points" without naming one
specific metric — checking all four independently is the more
conservative, more useful reading (a model could show low overfit on
accuracy alone while precision or recall for the hateful class drifts
much further, especially under class imbalance).

**Alternatives considered**: checking only accuracy (simplest) —
rejected, since accuracy is already known to be a weak signal for this
imbalanced dataset (spec 050's own research.md); checking only F1 —
rejected as still narrower than what "the gap between training and test
metrics" naturally reads as.

## Decision: load the persisted model/vectorizer read-only (no refit)

**Rationale**: EARS criterion 2 exists because re-fitting during
evaluation would silently produce a different model than the one spec
050 actually chose and persisted — the whole point of this task is to
evaluate *that specific artifact*, not a freshly retrained stand-in.

**Alternatives considered**: retrain inline inside
`scripts/evaluate_model.py` for convenience — rejected, defeats the
purpose of persisting `baseline_model.joblib`/`baseline_vectorizer.joblib`
in the first place.
