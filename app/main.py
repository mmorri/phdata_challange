from fastapi import Depends, FastAPI, HTTPException, status

from .models import (
    ErrorResponse,
    HouseFeatures,
    MinimalHouseFeatures,
    ModelInfo,
    PredictionResponse,
)
from .predictor import PredictionService
from .utils import get_settings

settings = get_settings()
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
)


def get_service() -> PredictionService:
    return PredictionService.get_instance()


@app.on_event("startup")
async def warm_model() -> None:
    get_service()


@app.get("/health", response_model=dict)
async def health() -> dict:
    return {"status": "ok"}


@app.get(
    "/model/info",
    response_model=ModelInfo,
    responses={status.HTTP_404_NOT_FOUND: {"model": ErrorResponse}},
)
async def model_info(service: PredictionService = Depends(get_service)) -> ModelInfo:
    info = service.model_metadata()
    if info is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model info unavailable")
    return info


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse}},
)
async def predict(
    payload: HouseFeatures,
    service: PredictionService = Depends(get_service),
) -> PredictionResponse:
    try:
        return service.predict(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post(
    "/predict/minimal",
    response_model=PredictionResponse,
    responses={status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse}},
)
async def predict_minimal(
    payload: MinimalHouseFeatures,
    service: PredictionService = Depends(get_service),
) -> PredictionResponse:
    try:
        return service.predict_minimal(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
