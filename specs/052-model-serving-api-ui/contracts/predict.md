# Contract: `POST /predict`

## Request

```
POST /predict
Content-Type: application/json

{"text": "some comment to classify", "language": "en"}
```

- `text`: string, required, non-empty (whitespace-only counts as empty).
- `language`: `"en"` or `"es"`, optional, defaults to `"en"`. Selects the
  cleaning/stopword/stemming pipeline
  (`specs/006-text-preprocessing/spec.md`) applied before vectorizing —
  required because the persisted vectorizer's vocabulary is stemmed,
  language-specific tokens (e.g. `"disgust"`/`"inmigr"`, not
  `"disgusting"`/`"immigrant"`), so skipping this step or using the
  wrong language silently produces near-random predictions.

## Response — 200 OK

```json
{
  "label": 1,
  "prediction": "hate",
  "probability": 0.87
}
```

- `label`: `0` or `1`.
- `prediction`: `"not_hate"` or `"hate"` (mirrors `label`).
- `probability`: float in `[0, 1]`, or `null` if the loaded model has no
  `predict_proba`.

## Response — 422 Unprocessable Entity

Returned when `text` is missing or empty. Body is FastAPI/Pydantic's
standard validation-error shape.

## Contract: `GET /health`

### Response — 200 OK

```json
{"status": "ok"}
```

No request body. Used to confirm the API process is up and the model
artifacts loaded successfully at startup.
