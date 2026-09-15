# Implementation Plan: Queryable interface for the model (API + UI)

**Branch**: `feature/model-serving-api-ui` | **Date**: 2026-09-15 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/052-model-serving-api-ui/spec.md`

## Summary

Build one shared prediction module (`src/models/predict.py`) that loads
the persisted model/vectorizer and turns raw text into a label, then
wire it into a thin FastAPI endpoint and a thin Streamlit app, so both
surfaces call the same code instead of each re-implementing inference.

## Technical Context

**Language/Version**: Python 3.14 (matches the rest of `src/`)

**Primary Dependencies**: FastAPI, `uvicorn` (ASGI server), Streamlit,
`pydantic` (FastAPI's request/response models, transitive dependency) —
all already in `requirements.txt`; joblib for loading the persisted
artifacts

**Storage**: reads `data/processed/baseline_model.joblib` and
`data/processed/baseline_vectorizer.joblib` (read-only); writes nothing

**Testing**: pytest + FastAPI's `TestClient` (httpx-based, bundled with
FastAPI) for the API; plain function calls for `predict_label`. Tests
inject a small fake fitted scikit-learn-like estimator/vectorizer via
`app.dependency_overrides`, never the real `.joblib` files — those are
gitignored (`specs/048-spanish-category-augmentation/spec.md`'s
precedent) and won't exist in the CI checkout
(`.github/workflows/ci.yml` only runs `pip install` + `pytest`, no data
pipeline step)

**Target Platform**: local — `uvicorn` dev server and `streamlit run`,
both run manually per README instructions (no deployment, per spec.md's
Out of scope)

**Project Type**: single Python library/CLI-script project, adding two
thin presentation layers (`src/api/`, `src/app/`) on top of the existing
`src/models/` package

**Performance Goals**: single-text prediction in well under a second
(TF-IDF transform + Logistic Regression predict on one row is
near-instant, same order of magnitude as the batch predictions already
measured in `specs/051-model-evaluation/spec.md`)

**Constraints**: the FastAPI app must not load/fit anything at import
time in a way that breaks test collection when the real `.joblib` files
are absent (CI) — artifact loading happens through an overridable
dependency, not a module-level side effect

**Scale/Scope**: single-endpoint API (`POST /predict`, `GET /health`),
single-page Streamlit app — no additional entities beyond what specs
050/051 already defined

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Practicality over precision**: a working FastAPI endpoint + Streamlit
  demo is exactly the "deployable, usable solution" the constitution
  asks for over a model that only exists as a file — pass.
- **Reproducibility**: both surfaces are scriptable/runnable via
  documented commands (`uvicorn ...`, `streamlit run ...`), not manual
  notebook steps — pass.
- **Incremental delivery**: scoped to serving only, no auth/deployment/
  persistence bundled in (all explicitly out of scope) — pass.
- **Documented code**: short docstrings matching
  `src/evaluation/evaluate.py`'s style — pass.
- No violations requiring the Complexity Tracking table.

## Project Structure

### Documentation (this feature)

```text
specs/052-model-serving-api-ui/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── contracts/             # Phase 1 output (this feature DOES expose an
│                          # external interface — the API)
├── quickstart.md         # Phase 1 output
└── spec.md               # Feature spec (already exists)
```

### Source Code (repository root)

```text
src/
├── models/
│   └── predict.py          # NEW: load_artifacts, predict_label
├── api/
│   ├── __init__.py          # already exists (empty)
│   └── main.py               # NEW: FastAPI app (POST /predict, GET /health)
└── app/
    ├── __init__.py           # already exists (empty)
    └── streamlit_app.py       # NEW: single-page Streamlit UI

tests/
├── test_predict.py            # NEW: unit tests, fake model/vectorizer
└── test_api.py                 # NEW: FastAPI TestClient, dependency override
```

**Structure Decision**: single-project layout, filling in the already-
reserved `src/api/` and `src/app/` packages
(`specs/003-project-structure-setup/spec.md`) alongside a new
`src/models/predict.py` next to the existing `src/models/train.py`.

## Complexity Tracking

No Constitution Check violations — table not needed.
