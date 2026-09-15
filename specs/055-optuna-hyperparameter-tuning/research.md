# Phase 0 Research: Hyperparameter tuning with Optuna

## Decision: tune `C`, `penalty`, `class_weight` only

**Rationale**: these three arguments are the ones most likely to move a
linear TF-IDF baseline's F1: `C` controls regularization strength
(under/over-fitting trade-off directly relevant after spec 051's
overfit check), `penalty` controls sparsity (`l1` can zero out
uninformative TF-IDF columns), `class_weight` controls how hard the
~63/37 imbalance is corrected for. Other `LogisticRegression` knobs
(`tol`, `max_iter`, `intercept_scaling`, ...) have much smaller expected
impact for this problem shape.

**Alternatives considered**: also searching over vectorizer
hyperparameters (`min_df`, `max_features`) — rejected as out of scope;
issue #14 says "hyperparameter search space **for the chosen model**",
and conflating vectorizer + model search spaces would roughly double
the trial cost for a Thursday-constrained pass.

## Decision: `solver="liblinear"`, fixed (not tuned)

**Rationale**: `liblinear` is the one scikit-learn solver that supports
both `l1` and `l2` penalty without extra constraints (unlike `lbfgs`,
`l2`-only) and performs well on medium-sized sparse TF-IDF data — no
need to add a `solver` axis to the search space just to unlock `l1`.

**Alternatives considered**: `saga` (also supports both penalties, plus
`elasticnet`) — rejected for this pass; `saga` typically needs more
iterations to converge on sparse high-dimensional data, adding runtime
for a search space that doesn't need `elasticnet`'s extra flexibility.

## Decision: cross-validate on the training split, evaluate once on test

**Rationale**: this is the whole point of the spec (EARS criteria 2-3)
— using the test split as a tuning signal, even indirectly (e.g. early
stopping trials based on test score), would invalidate
`specs/051-model-evaluation/spec.md`'s overfitting check by construction.
3-fold `StratifiedKFold` on the ~121k-row training split gives each fold
~40k rows, plenty for a stable F1 estimate without the cost of 5+ folds.

**Alternatives considered**: holding out a separate validation split
from training (train/val/test three-way split) instead of CV — rejected;
CV uses the training data more efficiently (no held-out validation rows
sitting idle) and is the more standard pairing with Optuna.

## Decision: `n_trials=20`

**Rationale**: with a 3-parameter space and 3-fold CV, 20 trials is
enough for Optuna's default TPE sampler to meaningfully explore the
space (60 total model fits) while finishing in minutes, not hours —
matching the Thursday deadline and the constitution's practicality
principle over a more exhaustive but slower search.

**Alternatives considered**: `n_trials=100`+ — more thorough, but not a
good time trade-off right now; the Result section will show the CV
score's trajectory, and this can be revisited later if the 20-trial
result looks like it hasn't converged.
