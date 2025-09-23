import json
import pathlib
import pickle
from typing import Dict, List, Tuple

import pandas as pd
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'
]
OUTPUT_DIR = "model"


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with demographics data
        sales_column_selection: list of columns from sales data to use

    Returns:
        Tuple with DataFrame of features and Series of target values
    """
    data = pd.read_csv(sales_path, usecols=sales_column_selection, dtype={'zipcode': str})
    demographics = pd.read_csv(demographics_path, dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left", on="zipcode")

    merged_data = merged_data.fillna(merged_data.median(numeric_only=True))

    y = merged_data.pop('price')
    x = merged_data.drop(columns=['zipcode'])

    return x, y


def main():
    """Load data, train model, and export artifacts."""
    print("Loading data...")
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    print(f"Data shape: {x.shape}")
    print(f"Features: {list(x.columns)}")

    print("Splitting data...")
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    candidate_specs = {
        "knn": {
            "display_name": "KNeighborsRegressor",
            "estimator": pipeline.make_pipeline(
                preprocessing.RobustScaler(),
                neighbors.KNeighborsRegressor()
            ),
            "param_grid": {
                'kneighborsregressor__n_neighbors': [3, 5, 7, 9, 11],
                'kneighborsregressor__weights': ['uniform', 'distance'],
                'kneighborsregressor__p': [1, 2],
            },
        },
        "random_forest": {
            "display_name": "RandomForestRegressor",
            "estimator": pipeline.Pipeline([
                ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
            ]),
            "param_grid": {
                'model__n_estimators': [200, 400],
                'model__max_depth': [None, 20, 40],
                'model__min_samples_leaf': [1, 2],
            },
        },
        "gradient_boosting": {
            "display_name": "GradientBoostingRegressor",
            "estimator": pipeline.Pipeline([
                ("model", GradientBoostingRegressor(random_state=42))
            ]),
            "param_grid": {
                'model__n_estimators': [200, 400],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 5],
            },
        },
        "hist_gradient_boosting": {
            "display_name": "HistGradientBoostingRegressor",
            "estimator": pipeline.Pipeline([
                ("model", HistGradientBoostingRegressor(random_state=42))
            ]),
            "param_grid": {
                'model__max_depth': [None, 10],
                'model__learning_rate': [0.05, 0.1],
                'model__max_leaf_nodes': [31, 63],
            },
        },
    }

    candidate_results = []
    best_candidate: Dict[str, object] | None = None
    best_validation_mae = float("inf")

    for key, spec in candidate_specs.items():
        print(f"\nEvaluating candidate: {spec['display_name']} ({key})")
        search = model_selection.GridSearchCV(
            spec["estimator"],
            spec["param_grid"],
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1,
        )
        search.fit(x_train, y_train)

        estimator = search.best_estimator_
        train_score = estimator.score(x_train, y_train)
        test_score = estimator.score(x_test, y_test)
        validation_mae = mean_absolute_error(y_test, estimator.predict(x_test))

        candidate_summary = {
            "name": key,
            "display_name": spec["display_name"],
            "best_params": search.best_params_,
            "cv_best_mae": float(-search.best_score_),
            "validation_mae": float(validation_mae),
            "train_score": float(train_score),
            "test_score": float(test_score),
        }
        candidate_results.append({**candidate_summary, "estimator": estimator})

        print(f"Best params: {search.best_params_}")
        print(f"Best CV MAE: {candidate_summary['cv_best_mae']:.2f}")
        print(f"Validation MAE: {validation_mae:,.2f}")
        print(f"Test R² score: {test_score:.4f}")

        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_candidate = {
                "key": key,
                "summary": candidate_summary,
                "estimator": estimator,
            }

    assert best_candidate is not None, "At least one candidate must be evaluated"

    model = best_candidate["estimator"]
    summary = best_candidate["summary"]
    print("\nSelected model:", summary["display_name"])
    print("Validation MAE:", f"{summary['validation_mae']:,.2f}")
    print("Test R² score:", f"{summary['test_score']:.4f}")

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    print(f"Saving model artifacts to {output_dir}/...")
    pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
    json.dump(list(x_train.columns), open(output_dir / "model_features.json", 'w'))

    candidate_payload = []
    for entry in candidate_results:
        payload = {k: v for k, v in entry.items() if k != "estimator"}
        candidate_payload.append(payload)

    model_info = {
        "model_type": summary["display_name"],
        "version": "1.0",
        "features": list(x_train.columns),
        "n_features": len(x_train.columns),
        "train_score": float(summary["train_score"]),
        "test_score": float(summary["test_score"]),
        "training_samples": len(x_train),
        "best_params": summary["best_params"],
        "cv_best_mae": float(summary["cv_best_mae"]),
        "validation_mae": float(summary["validation_mae"]),
        "candidates": candidate_payload,
    }
    json.dump(model_info, open(output_dir / "model_info.json", 'w'), indent=2)

    print("Model training complete!")


if __name__ == "__main__":
    main()
