from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .utils import get_logger

LOGGER = get_logger(__name__)


class DemographicsLoader:
    """Load and cache demographics data for zipcode enrichment."""

    def __init__(self, path: Path):
        self._path = path
        self._data: Optional[pd.DataFrame] = None
        self._medians: Optional[pd.Series] = None

    def load(self) -> pd.DataFrame:
        if self._data is None:
            LOGGER.info("Loading demographics data from %s", self._path)
            dataframe = pd.read_csv(self._path, dtype={"zipcode": str})
            dataframe = dataframe.set_index("zipcode")
            self._data = dataframe
            self._medians = dataframe.median(numeric_only=True)
        return self._data

    def get_by_zipcode(self, zipcode: str) -> Optional[pd.Series]:
        dataframe = self.load()
        if zipcode in dataframe.index:
            return dataframe.loc[zipcode]
        return None

    def fallback_values(self) -> Dict[str, float]:
        if self._medians is None:
            self.load()
        assert self._medians is not None
        return self._medians.to_dict()


def build_demographics_loader(path: Path) -> DemographicsLoader:
    return DemographicsLoader(path)
