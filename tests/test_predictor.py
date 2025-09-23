from pathlib import Path

import pandas as pd
import pytest

from app.models import HouseFeatures, MinimalHouseFeatures
from app.predictor import PredictionService

MODEL_PATH = Path("model/model.pkl")
SAMPLE_DATA = Path("data/future_unseen_examples.csv")


@pytest.fixture(scope="module")
def service() -> PredictionService:
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact missing; run scripts/train_model.py")
    return PredictionService.get_instance()


@pytest.fixture(scope="module")
def sample_row() -> dict:
    if not SAMPLE_DATA.exists():
        pytest.skip("Sample data not available")
    dataframe = pd.read_csv(SAMPLE_DATA, dtype={"zipcode": str})
    return dataframe.iloc[0].to_dict()


def test_predict_full(service: PredictionService, sample_row: dict) -> None:
    payload = {k: v for k, v in sample_row.items() if k != "price"}
    request = HouseFeatures(**payload)
    response = service.predict(request)
    assert response.prediction > 0
    assert response.zipcode == request.zipcode


def test_predict_minimal(service: PredictionService, sample_row: dict) -> None:
    payload = {k: sample_row[k] for k in MinimalHouseFeatures.__fields__.keys() if k in sample_row}
    request = MinimalHouseFeatures(**payload)
    response = service.predict_minimal(request)
    assert response.prediction > 0
    assert response.zipcode == request.zipcode
