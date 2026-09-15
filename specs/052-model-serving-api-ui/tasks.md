# Tasks: Queryable interface for the model (API + UI)

**Input**: Design documents from `/specs/052-model-serving-api-ui/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md,
contracts/predict.md, quickstart.md

**Tests**: included — this repo tests every `src/` module.

**Organization**: single cohesive story (US1) — shared prediction
module, then the two thin surfaces that call it.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [x] T001 Create `specs/052-model-serving-api-ui/` spec, plan, research,
      data-model, contracts, quickstart (this session, prior steps)
- [x] T002 Confirm `src/api/__init__.py` and `src/app/__init__.py`
      already exist (they do, from initial scaffolding) — no action
      needed

## Phase 2: Foundational

No new foundational infrastructure needed — reuses the persisted
artifacts from `specs/050-baseline-model-training/spec.md`.

## Phase 3: User Story 1 - Predict via API and via a manual-testing UI, sharing one prediction path (Priority: P1) 🎯 MVP

**Goal**: a shared `predict_label` function used by both a FastAPI
endpoint and a Streamlit app, so a user can submit text through either
surface and get a hate/not-hate prediction.

**Independent Test**: `curl` the running API and confirm the JSON shape
in `contracts/predict.md`; run the Streamlit app and confirm a submitted
comment shows a prediction in-page; run
`pytest tests/test_predict.py tests/test_api.py` against fake
model/vectorizer fixtures.

### Tests for User Story 1

- [x] T003 [P] [US1] Write `tests/test_predict.py`: a small fake fitted
      vectorizer (`CountVectorizer` or `TfidfVectorizer` fit on a tiny
      corpus, e.g. `["hate you", "nice day"]`) and a small fake fitted
      `LogisticRegression` on 2-3 rows; assert `predict_label()` returns
      `label` in `{0, 1}`, `prediction` matching `label`
      (`"hate"`/`"not_hate"`), and a `probability` in `[0, 1]`; assert
      that a fake estimator without `predict_proba` (e.g. a bare object
      exposing only `.predict()`) yields `probability=None` instead of
      raising
- [x] T004 [P] [US1] Write `tests/test_api.py`: build the FastAPI app,
      override its predictor dependency with a fake `predict_label`
      result via `app.dependency_overrides`, use `TestClient` to assert
      `POST /predict` with a valid body returns 200 and the shape from
      `contracts/predict.md`, `POST /predict` with an empty/missing
      `text` returns 422, and `GET /health` returns
      `{"status": "ok"}`

### Implementation for User Story 1

- [x] T005 [US1] Implement `src/models/predict.py`:
      `load_artifacts(model_path, vectorizer_path)` (returns
      `(model, vectorizer)` via `joblib.load`); `predict_label(text,
      model, vectorizer)` (returns `{"label", "prediction",
      "probability"}` per `data-model.md`, `probability=None` when the
      model has no `predict_proba`) (depends on T003 existing first so
      the tests fail red before this lands)
- [x] T006 [US1] Implement `src/api/main.py`: FastAPI app, a
      `get_predictor()` dependency that calls `load_artifacts()` with
      the default `data/processed/baseline_model.joblib` /
      `baseline_vectorizer.joblib` paths (cached, loaded once), `POST
      /predict` (Pydantic request model requiring non-empty `text`,
      calls `predict_label`) and `GET /health` per
      `contracts/predict.md` (depends on T004 existing first, and on
      T005 for `predict_label`)
- [x] T007 [US1] Implement `src/app/streamlit_app.py`: text input +
      "Predict" button, calls `load_artifacts()` once (cached via
      `st.cache_resource`) and `predict_label()` directly, displays
      `prediction` (and `probability` when not `None`) (depends on T005)
- [x] T008 [US1] Manually run the API (`uvicorn`) and the Streamlit app
      per `quickstart.md` against the real persisted artifacts, confirm
      both work end-to-end, and fill in spec.md's Result section
      (depends on T006, T007)

**Checkpoint**: User Story 1 (the whole feature) is functional and
independently testable.

## Phase 4: Polish & Cross-Cutting Concerns

- [x] T009 [P] Update `specs/001-hate-speech-detection/tasks.md` section 7
      checkboxes to `[x]` once T003-T008 are done
- [x] T010 [P] Add API/Streamlit run instructions to README.md's
      pipeline section
- [x] T011 Run `pytest` (full suite) and confirm no regressions
- [x] T012 Run `quickstart.md` end-to-end as a final sanity check

## Dependencies & Execution Order

- Setup (T001-T002) → done already, no blockers
- Foundational: none needed, US1 can start immediately
- Within US1: T003, T004 (tests, parallel — different files) before T005
  (implementation, red-first) before T006/T007 (the two surfaces, both
  depend on T005 but are independent of each other) before T008 (manual
  real-artifact run, depends on both)
- Polish (T009-T012) depends on US1 (T003-T008) being complete

## Implementation Strategy

### MVP First (User Story 1 Only)

1. T003, T004: write both test files against fakes, confirm they fail
   (no `src/models/predict.py` / `src/api/main.py` yet)
2. T005: implement `src/models/predict.py`, confirm T003 passes
3. T006: implement `src/api/main.py`, confirm T004 passes
4. T007: implement `src/app/streamlit_app.py`
5. T008: run both surfaces for real, record the Result in spec.md
6. **STOP and VALIDATE**: `pytest tests/test_predict.py tests/test_api.py -q`
   passes

### Incremental Delivery

This feature *is* the MVP — there is only one story. Polish (T009-T012)
wraps it up: update the master tasks.md, document it in README, run the
full test suite, and do a final quickstart pass before opening the PR.
