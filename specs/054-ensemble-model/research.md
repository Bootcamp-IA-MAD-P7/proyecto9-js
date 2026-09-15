# Phase 0 Research: Ensemble model

## Decision: hard voting, not soft voting

**Rationale**: scikit-learn's `VotingClassifier(voting="soft")` requires
every estimator to expose `predict_proba`. `LinearSVC` (one of the three
baseline algorithms) does not by default — only `decision_function`.
Enabling soft voting would need wrapping it in `CalibratedClassifierCV`,
adding real training-time cost and complexity for a first ensemble pass.
Hard voting (majority label vote across the three) needs no such
wrapper, reuses `build_models()` unchanged, and is the simpler technique
issue #11 lists first ("Voting/Stacking classifier").

**Alternatives considered**: soft voting via `CalibratedClassifierCV`
around `LinearSVC` — deferred; worth revisiting only if hard voting's
result motivates the extra complexity. Random Forest (issue #11's other
named example) — rejected as the primary approach here since it's a
different algorithm family from anything already built, whereas voting
over the three existing baselines directly answers "does combining what
we have help" with zero new model-type risk.

## Decision: persist under new filenames, never touch the served baseline

**Rationale**: `specs/052-model-serving-api-ui/spec.md`'s API/Streamlit
load `data/processed/baseline_model.joblib` by fixed path. Overwriting
it here would silently change what's served based on a single
comparison run, without the explicit review a model swap deserves.
Persisting to `ensemble_model.joblib`/`ensemble_vectorizer.joblib`
keeps both artifacts available for a deliberate follow-up decision.

**Alternatives considered**: overwrite `baseline_model.joblib` if the
ensemble wins — rejected as too automatic a decision for something this
consequential; matches the same caution spec 052's research.md applied
to model loading (read-only, no silent side effects).
