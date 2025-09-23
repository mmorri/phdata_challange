from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import MinimalHouseFeatures

MODEL_PATH = Path("model/model.pkl")
SAMPLE_DATA = Path("data/future_unseen_examples.csv")


@pytest.fixture(scope="module")
def client() -> TestClient:
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact missing; run scripts/train_model.py")
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_row() -> dict:
    if not SAMPLE_DATA.exists():
        pytest.skip("Sample data not available")
    dataframe = pd.read_csv(SAMPLE_DATA, dtype={"zipcode": str})
    return dataframe.iloc[0].to_dict()


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint(client: TestClient, sample_row: dict) -> None:
    payload = {k: v for k, v in sample_row.items() if k != "price"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert body["demographic_data_found"] in {True, False}


def test_predict_minimal_endpoint(client: TestClient, sample_row: dict) -> None:
    payload = {k: sample_row[k] for k in MinimalHouseFeatures.__fields__.keys() if k in sample_row}
    response = client.post("/predict/minimal", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert body["zipcode"] == payload["zipcode"].zfill(5)
