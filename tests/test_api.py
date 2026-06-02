from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "neighborhood": "Koreatown",
    "primary_type": "korean_restaurant",
    "price_level": 2,
    "review_count": 500,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready():
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}


def test_meta_returns_lists():
    r = client.get("/meta")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["neighborhoods"], list)
    assert len(body["neighborhoods"]) > 0
    assert isinstance(body["place_types"], list)


def test_predict_valid():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "high_rated_probability" in body
    assert 0.0 <= body["high_rated_probability"] <= 1.0
    assert isinstance(body["predicted_high_rated"], bool)


def test_predict_missing_field():
    r = client.post("/predict", json={"neighborhood": "Koreatown"})
    assert r.status_code == 422


def test_predict_invalid_price_level():
    payload = {**VALID_PAYLOAD, "price_level": 9}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_negative_reviews():
    payload = {**VALID_PAYLOAD, "review_count": -1}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
