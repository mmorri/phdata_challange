import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application configuration sourced from environment variables."""

    model_dir: Path = Path("./model")
    model_filename: str = "model.pkl"
    model_features_file: str = "model_features.json"
    model_info_file: str = "model_info.json"
    sales_data_path: Path = Path("data/kc_house_data.csv")
    demographics_path: Path = Path("data/zipcode_demographics.csv")
    api_title: str = "Sound Realty Price Service"
    api_version: str = "1.0.0"
    api_description: str = "REST API for house price predictions."

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(base: Path, filename: str) -> Path:
    path = base / filename
    return path.resolve()
