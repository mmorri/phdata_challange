from __future__ import annotations

import pickle
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .data_loader import DemographicsLoader, build_demographics_loader
from .models import (
    HouseFeatures,
    MinimalHouseFeatures,
    ModelInfo,
    PredictionMetadata,
    PredictionResponse,
)
from .utils import (
    get_logger,
    get_settings,
    load_json,
    resolve_path,
    utc_timestamp,
)

LOGGER = get_logger(__name__)
HOUSE_FEATURE_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]


class PredictionService:
    _instance: Optional["PredictionService"] = None
    _lock = Lock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model: BaseEstimator = self._load_model()
        self.model_features = self._load_model_features()
        self.model_info = self._load_model_info()
        self.demographics_loader: DemographicsLoader = build_demographics_loader(
            self.settings.demographics_path
        )
        self.house_feature_defaults = self._load_house_feature_defaults()
        fallback_demographics = self.demographics_loader.fallback_values()
        default_features: Dict[str, float] = {}
        for feature in self.model_features:
            if feature in self.house_feature_defaults:
                default_features[feature] = float(self.house_feature_defaults[feature])
            elif feature in fallback_demographics:
                default_features[feature] = float(fallback_demographics[feature])
            else:
                default_features[feature] = 0.0
        self._feature_defaults = default_features

    @classmethod
    def get_instance(cls) -> "PredictionService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def predict(self, payload: HouseFeatures) -> PredictionResponse:
        payload_dict = payload.dict()
        return self._run_inference(payload_dict)

    def predict_minimal(self, payload: MinimalHouseFeatures) -> PredictionResponse:
        payload_dict = self.house_feature_defaults.copy()
        payload_values = payload.dict()
        zipcode = payload_values["zipcode"]
        payload_dict.update(payload_values)
        payload_dict["zipcode"] = zipcode
        return self._run_inference(payload_dict)

    def model_metadata(self) -> Optional[ModelInfo]:
        return self.model_info

    def _run_inference(self, payload: Dict[str, Any]) -> PredictionResponse:
        zipcode = payload.get("zipcode")
        if not zipcode:
            raise ValueError("zipcode is required for prediction")

        features_frame, demographic_found = self._build_feature_frame(payload)
        prediction = float(self.model.predict(features_frame)[0])
        confidence_interval = self._confidence_interval(features_frame)

        metadata = PredictionMetadata(
            model_type=self.model_info.model_type if self.model_info else type(self.model).__name__,
            confidence_interval=confidence_interval,
            neighbors_considered=self._neighbors_considered(),
        )
        response = PredictionResponse(
            prediction=prediction,
            model_version=self.model_info.version if self.model_info else "unknown",
            zipcode=zipcode,
            demographic_data_found=demographic_found,
            features_used=len(self.model_features),
            timestamp=utc_timestamp(),
            metadata=metadata,
        )
        return response

    def _build_feature_frame(self, payload: Dict[str, Any]) -> Tuple[pd.DataFrame, bool]:
        zipcode = payload.get("zipcode")
        base_features = {
            feature: payload[feature]
            for feature in HOUSE_FEATURE_COLUMNS
            if feature in payload
        }
        demographics_series = self.demographics_loader.get_by_zipcode(zipcode)  # type: ignore[arg-type]
        demographic_found = demographics_series is not None
        if demographics_series is None:
            demographics_data = self.demographics_loader.fallback_values()
        else:
            demographics_data = demographics_series.to_dict()

        combined_features: Dict[str, Any] = {**self._feature_defaults}
        combined_features.update(base_features)
        combined_features.update(demographics_data)

        row = pd.DataFrame([combined_features])
        row = row[list(self.model_features)].astype(float)
        return row, demographic_found

    def _load_model(self) -> BaseEstimator:
        settings = self.settings
        model_path = resolve_path(settings.model_dir, settings.model_filename)
        LOGGER.info("Loading model from %s", model_path)
        # Pickle can execute arbitrary code: refuse artifacts whose SHA-256
        # does not match MODEL_SHA256 when it is pinned (set it in CI after
        # training). Regenerate artifacts with create_model.py otherwise.
        import hashlib
        import io
        import os

        data = model_path.read_bytes()
        expected = os.environ.get("MODEL_SHA256", "")
        if expected:
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected.lower():
                raise ValueError(f"{model_path.name}: SHA-256 mismatch — refusing to unpickle")
        return pickle.load(io.BytesIO(data))

    def _load_model_features(self) -> Tuple[str, ...]:
        settings = self.settings
        feature_path = resolve_path(settings.model_dir, settings.model_features_file)
        LOGGER.info("Loading model feature order from %s", feature_path)
        features = load_json(feature_path)
        return list(features)

    def _load_model_info(self) -> Optional[ModelInfo]:
        settings = self.settings
        info_path = resolve_path(settings.model_dir, settings.model_info_file)
        if not info_path.exists():
            LOGGER.warning("Model info file not found at %s", info_path)
            return None
        info_dict = load_json(info_path)
        return ModelInfo(**info_dict)

    def _load_house_feature_defaults(self) -> Dict[str, float]:
        settings = self.settings
        columns = HOUSE_FEATURE_COLUMNS + ["zipcode"]
        try:
            dataframe = pd.read_csv(
                settings.sales_data_path,
                usecols=columns,
                dtype={"zipcode": str},
            )
        except FileNotFoundError:
            LOGGER.warning("Sales data not found at %s; falling back to zeros", settings.sales_data_path)
            return {feature: 0.0 for feature in HOUSE_FEATURE_COLUMNS}
        medians = dataframe[HOUSE_FEATURE_COLUMNS].median(numeric_only=True)
        return medians.to_dict()

    def _neighbors_considered(self) -> Optional[int]:
        try:
            pipeline = self.model
            regressor = getattr(pipeline, "named_steps", {}).get("kneighborsregressor")
            if regressor is None:
                return None
            return int(getattr(regressor, "n_neighbors", 0)) or None
        except AttributeError:
            return None

    def _confidence_interval(self, features: pd.DataFrame) -> Optional[list]:
        try:
            pipeline = self.model
            named_steps = getattr(pipeline, "named_steps", {})
            scaler = named_steps.get("robustscaler")
            regressor = named_steps.get("kneighborsregressor")
            if scaler is None or regressor is None:
                return None
            transformed = scaler.transform(features)
            distances, indices = regressor.kneighbors(transformed, return_distance=True)
            neighbor_targets = regressor._y[indices[0]]
            lower = float(np.percentile(neighbor_targets, 5))
            upper = float(np.percentile(neighbor_targets, 95))
            return [lower, upper]
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Unable to compute confidence interval: %s", exc)
            return None
