from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class HouseFeatures(BaseModel):
    bedrooms: int = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)
    sqft_living: int = Field(..., ge=0)
    sqft_lot: int = Field(..., ge=0)
    floors: float = Field(..., ge=0)
    waterfront: int = Field(..., ge=0)
    view: int = Field(..., ge=0)
    condition: int = Field(..., ge=1)
    grade: int = Field(..., ge=1)
    sqft_above: int = Field(..., ge=0)
    sqft_basement: int = Field(..., ge=0)
    yr_built: int = Field(..., ge=1800)
    yr_renovated: int = Field(..., ge=0)
    zipcode: str
    lat: float
    long: float
    sqft_living15: int = Field(..., ge=0)
    sqft_lot15: int = Field(..., ge=0)

    @field_validator("zipcode")
    @classmethod
    def zipcode_to_string(cls, value: str) -> str:
        return value.zfill(5)


class MinimalHouseFeatures(BaseModel):
    bedrooms: int = Field(..., ge=0)
    bathrooms: float = Field(..., ge=0)
    sqft_living: int = Field(..., ge=0)
    sqft_lot: int = Field(..., ge=0)
    floors: float = Field(..., ge=0)
    sqft_above: int = Field(..., ge=0)
    sqft_basement: int = Field(..., ge=0)
    zipcode: str

    @field_validator("zipcode")
    @classmethod
    def minimal_zipcode(cls, value: str) -> str:
        return value.zfill(5)


class PredictionMetadata(BaseModel):
    model_type: str
    confidence_interval: Optional[List[float]] = None
    neighbors_considered: Optional[int] = None


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    zipcode: str
    demographic_data_found: bool
    features_used: int
    timestamp: str
    metadata: PredictionMetadata


class ErrorResponse(BaseModel):
    detail: str


class ModelCandidate(BaseModel):
    name: str
    display_name: str
    best_params: Dict[str, Any]
    cv_best_mae: float
    validation_mae: float
    train_score: float
    test_score: float


class ModelInfo(BaseModel):
    model_type: str
    version: str
    features: List[str]
    n_features: int
    train_score: Optional[float] = None
    test_score: Optional[float] = None
    training_samples: Optional[int] = None
    best_params: Optional[Dict[str, Any]] = None
    cv_best_mae: Optional[float] = None
    validation_mae: Optional[float] = None
    candidates: Optional[List[ModelCandidate]] = None
