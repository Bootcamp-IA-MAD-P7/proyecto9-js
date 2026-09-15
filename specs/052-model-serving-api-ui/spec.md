# Spec 052: Queryable interface for the model (API + UI)

## Linked issue

- GitHub issue: #9
- Kanban status at spec creation time: In Progress
- Level: Nivel Esencial

## What

Expose the persisted baseline model
(`specs/050-baseline-model-training/spec.md`,
`specs/051-model-evaluation/spec.md`) through a FastAPI endpoint that
accepts raw text and returns a hate/not-hate prediction, and a Streamlit
app for manual testing — both driven by one shared prediction module, so
the logic isn't duplicated between them.

## Why

`specs/001-hate-speech-detection/tasks.md` section 7 ("Serving") is the
next unblocked task now that a model is trained
(`specs/050-baseline-model-training/spec.md`) and evaluated
(`specs/051-model-evaluation/spec.md`, confirmed not overfit). Issue #9
names both a FastAPI endpoint and a Streamlit app explicitly (not an
either/or), matching the constitution's "practicality over precision"
goal of a deployable, actually-queryable solution rather than a model
that only exists as a `.joblib` file.

## Scope

- `src/models/predict.py`:
  - `load_artifacts(model_path, vectorizer_path)`: loads the persisted
    model + vectorizer via `joblib.load` (read-only, matching
    `specs/051-model-evaluation/spec.md`'s no-refit rule).
  - `predict_label(text, model, vectorizer)`: transforms one raw text
    string and returns `{"label": 0|1, "prediction": "hate"|"not_hate",
    "probability": float|None}` — probability is `None` when the model
    has no `predict_proba` (not all scikit-learn classifiers expose one;
    Logistic Regression, the current persisted model, does).
- `src/api/main.py`: FastAPI app with `POST /predict` (body: `{"text":
  str}`, response: the `predict_label` dict) and `GET /health` (basic
  liveness check). Loads the real artifacts once via a dependency
  function that tests can override.
- `src/app/streamlit_app.py`: single text box + button UI calling
  `predict_label` directly (in-process function call, not an HTTP
  request to the API — both surfaces share the same underlying module,
  which is what "not duplicated" means here), displaying the prediction
  in the same session.
- `tests/test_predict.py`, `tests/test_api.py`: unit/integration tests
  using a small fake fitted model + vectorizer (never the real
  `data/processed/*.joblib` artifacts, which are gitignored and won't
  exist in CI — see `specs/048-spanish-category-augmentation/spec.md`'s
  precedent of keeping large data files out of git).

## EARS criteria

1. WHEN a raw text string is submitted, THE SYSTEM SHALL apply the same
   cleaning/tokenization/stemming pipeline
   (`specs/006-text-preprocessing/spec.md`) the model was trained on
   before vectorizing, and return a hate/not-hate prediction derived
   from the persisted baseline model, without refitting or modifying
   it.
2. WHEN the FastAPI endpoint receives a request, THE SYSTEM SHALL
   validate that `text` is present and non-empty, returning a 422/400
   response otherwise (not a 500).
3. WHEN the Streamlit app makes a prediction, THE SYSTEM SHALL use the
   exact same `predict_label` function the API uses — no separate
   prediction code path.
4. THE SYSTEM SHALL expose `load_artifacts` and `predict_label` as
   functions accepting model/vectorizer objects directly, so tests can
   inject a small fake fitted estimator instead of depending on the real
   `.joblib` files (which aren't committed to git and won't exist in the
   CI environment).
5. WHEN a user submits a comment through the Streamlit app, THE SYSTEM
   SHALL display the predicted label within the same session (per
   `specs/001-hate-speech-detection/tasks.md` section 7's own acceptance
   criterion).

## Result

Ran both surfaces manually against the real persisted artifacts.

**API** (`uvicorn src.api.main:app`): `GET /health` returned
`{"status": "ok"}`; `POST /predict` correctly classified both English
and Spanish examples once the preprocessing-mismatch bug below was
fixed; an empty `text` correctly returned `422`.

**Streamlit** (`streamlit run src/app/streamlit_app.py`): loaded, a
submitted English hateful comment ("you are all disgusting immigrants,
go back to your country") correctly showed **"Prediction: HATE"** with
**"Confidence: 55.1%"**, matching the API's response for the same text —
confirming both surfaces genuinely share `predict_label`.

**Bug found and fixed during manual testing:** the first implementation
of `predict_label` fed raw, unprocessed text straight into the
vectorizer's `.transform()`. Since the persisted vectorizer's vocabulary
is stemmed tokens from `clean_comment` (e.g. `"disgust"`, `"inmigr"`,
not `"disgusting"`, `"immigrant"`), this silently produced near-random
predictions — e.g. "you are all disgusting immigrants, go back to your
country" was misclassified as `not_hate` (28% confidence) before the
fix. Added a `language` parameter (default `"en"`, `"es"` also
supported) and now run `src/preprocessing/pipeline.py::preprocess` on
the input before vectorizing, matching training exactly. After the fix,
the same English example correctly returns `hate` (55.1%), and Spanish
examples ("eres una basura... maldito inmigrante" → `hate`, 96.7%; "que
tengas un lindo dia... gracias" → `not_hate`, 3.7%) are also classified
correctly. This is documented as a functional requirement, not a
one-off fix — see EARS criterion 1 and `contracts/predict.md`.

## Out of scope

- Authentication/rate limiting on the API — not named in issue #9, and
  this is a local/demo-scoped deliverable for the Nivel Esencial level.
- Persisting prediction results in a database — issue #20, a separate
  later/advanced task.
- Deploying the API/app to a public server — issue #17, a separate
  later/advanced task.
- Batch prediction (multiple comments at once) — issue #9 asks for "a
  user... submit a message" (singular), matching
  `specs/001-hate-speech-detection/tasks.md` section 7's wording.
