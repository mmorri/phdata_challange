import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Ensure repository root is importable when executed as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_model import (
    DEMOGRAPHICS_PATH,
    OUTPUT_DIR,
    SALES_COLUMN_SELECTION,
    SALES_PATH,
    load_data,
)


def evaluate() -> None:
    features, target = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model_path = Path(OUTPUT_DIR) / "model.pkl"
    with model_path.open("rb") as file:
        model = pickle.load(file)  # type: ignore[name-defined]

    predictions = model.predict(x_test)

    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    errors = y_test - predictions
    summary = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "error_std": float(np.std(errors)),
    }

    output_path = Path(OUTPUT_DIR) / "model_evaluation.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    evaluate()
