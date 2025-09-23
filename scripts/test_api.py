import argparse
import time
from typing import Dict, List

import pandas as pd
import requests

MINIMAL_FEATURES: List[str] = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "sqft_above",
    "sqft_basement",
    "zipcode",
]


def load_examples(path: str, limit: int) -> List[Dict[str, object]]:
    dataframe = pd.read_csv(path, dtype={"zipcode": str})
    dataframe = dataframe.head(limit)
    records = dataframe.to_dict(orient="records")
    return records


def invoke_predictions(base_url: str, examples: List[Dict[str, object]]) -> None:
    for idx, example in enumerate(examples, start=1):
        example_payload = {k: v for k, v in example.items() if k != "price"}
        start = time.perf_counter()
        response = requests.post(f"{base_url}/predict", json=example_payload, timeout=10)
        latency_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        print(f"Example {idx} full payload | {latency_ms:.2f} ms | {response.json()}")

        minimal_payload = {k: example_payload[k] for k in MINIMAL_FEATURES if k in example_payload}
        start = time.perf_counter()
        response = requests.post(f"{base_url}/predict/minimal", json=minimal_payload, timeout=10)
        latency_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        print(f"Example {idx} minimal payload | {latency_ms:.2f} ms | {response.json()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise the prediction API with sample data")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for the API service")
    parser.add_argument("--examples", default="data/future_unseen_examples.csv", help="CSV with sample requests")
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to send")
    args = parser.parse_args()

    examples = load_examples(args.examples, args.limit)
    invoke_predictions(args.base_url.rstrip("/"), examples)


if __name__ == "__main__":
    main()
