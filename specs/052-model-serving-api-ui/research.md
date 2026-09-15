# Phase 0 Research: Queryable interface for the model (API + UI)

## Decision: FastAPI dependency injection for the loaded model, not a module-level global

**Rationale**: loading `data/processed/baseline_model.joblib` at import
time would make `src/api/main.py` fail to even import in CI, since that
file is gitignored and never committed
(`specs/048-spanish-category-augmentation/spec.md`'s established
convention for `data/processed/*`). Using a FastAPI dependency
(`Depends(get_predictor)`) lets `tests/test_api.py` override it via
`app.dependency_overrides[get_predictor] = lambda: fake_predictor`
without touching disk at all.

**Alternatives considered**: lazy-load on first request with a
module-level cache (`if _model is None: _model = joblib.load(...)`) —
works too, but doesn't give tests a clean injection point; would need
monkeypatching `joblib.load` instead, which is more brittle than
FastAPI's built-in override mechanism.

## Decision: `predict_label` takes the model/vectorizer as parameters, not paths

**Rationale**: mirrors `specs/051-model-evaluation/spec.md`'s
`compute_metrics(y_true, y_pred)` pattern — plain functions over
in-memory objects are trivial to unit test with small fakes. Path-based
loading is isolated into the separate `load_artifacts()` function, called
exactly once (by the FastAPI dependency and by the Streamlit script), so
the expensive I/O happens once per process, not per request/prediction.

**Alternatives considered**: one combined `predict_from_paths(text,
model_path, vectorizer_path)` function that loads and predicts in one
call — rejected, it would reload the `.joblib` files from disk on every
single prediction, wasteful for a live API server, and harder to test
without real files on disk.

## Decision: Streamlit calls `predict_label` in-process, not via HTTP to the API

**Rationale**: spec.md's requirement is that the two surfaces "share the
same underlying module," not that one is a client of the other. Calling
the FastAPI server from Streamlit would mean running two processes and
handling the API being down as a UI-level error case — unnecessary
complexity for a Nivel Esencial "manual testing" tool. Importing
`src.models.predict` directly keeps the Streamlit app a single,
self-contained process while still guaranteeing byte-for-byte identical
prediction logic to the API (same function, same code path).

**Alternatives considered**: Streamlit app as an HTTP client of the
FastAPI server — more realistic for a "real" microservice split, but
explicitly heavier than what issue #9 asks for (manual testing tool);
revisit if/when issue #17 (deployment) makes the API a genuinely
separate long-running service other things need to call.

## Decision: `probability` is `None` when the model has no `predict_proba`

**Rationale**: the persisted baseline is Logistic Regression (has
`predict_proba`), but `predict_label` shouldn't assume that forever — a
future re-run of `scripts/train_baseline_model.py` could persist
`LinearSVC` instead (no `predict_proba` by default), and the API/UI
should degrade gracefully (label only) rather than crash.

**Alternatives considered**: always require `predict_proba` and raise if
missing — rejected as too brittle given `specs/050-baseline-model-training/
spec.md` picks the winner dynamically; a `None` probability is a much
smaller blast radius than an API that can start 500ing after a routine
retrain.
