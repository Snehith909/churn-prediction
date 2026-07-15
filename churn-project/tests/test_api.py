import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


@pytest.fixture
def valid_payload():
    return {
        "CreditScore": 619, "Geography": "France", "Gender": "Female",
        "Age": 42, "Tenure": 2, "Balance": 0, "NumOfProducts": 1,
        "HasCrCard": 1, "IsActiveMember": 1, "EstimatedSalary": 101348.88,
    }


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_payload_returns_200(valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert "will_churn" in body
    assert 0 <= body["churn_probability"] <= 1


def test_predict_rejects_invalid_age(valid_payload):
    valid_payload["Age"] = 5  # below min age of 18
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422  # pydantic validation error


def test_predict_rejects_missing_field(valid_payload):
    del valid_payload["CreditScore"]
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_rejects_invalid_credit_score_range(valid_payload):
    valid_payload["CreditScore"] = 100  # below min of 300
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422
