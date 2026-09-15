from fastapi.testclient import TestClient

from src.api.main import app, get_predictor


def _fake_predictor():
    def predict(text: str, language: str) -> dict:
        return {"label": 1, "prediction": "hate", "probability": 0.87}

    return predict


def test_health_check():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_expected_shape():
    app.dependency_overrides[get_predictor] = _fake_predictor
    client = TestClient(app)

    response = client.post("/predict", json={"text": "you are all disgusting"})

    assert response.status_code == 200
    assert response.json() == {"label": 1, "prediction": "hate", "probability": 0.87}
    app.dependency_overrides.clear()


def test_predict_rejects_empty_text():
    app.dependency_overrides[get_predictor] = _fake_predictor
    client = TestClient(app)

    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 422
    app.dependency_overrides.clear()


def test_predict_rejects_missing_text():
    app.dependency_overrides[get_predictor] = _fake_predictor
    client = TestClient(app)

    response = client.post("/predict", json={})

    assert response.status_code == 422
    app.dependency_overrides.clear()
