# API Reference

## Base URL
`http://{host}:{port}`

## GET /health
- **Purpose:** Service liveness probe
- **Response:** 200 OK `{ "status": "ok" }`

## GET /model/info
- **Purpose:** Retrieve model metadata
- **Response:** 200 OK with training metadata or 404 if unavailable
- **Sample Response:**
```json
{
  "model_type": "KNeighborsRegressor",
  "version": "1.0",
  "features": ["bedrooms", "bathrooms", "sqft_living", "median_income", "population", "median_age"],
  "n_features": 54,
  "train_score": 0.91,
  "test_score": 0.72,
  "training_samples": 17289
}
```

## POST /predict
- **Purpose:** Predict house price using the full feature set from `future_unseen_examples.csv`
- **Request Body:**
```json
{
  "bedrooms": 3,
  "bathrooms": 2.5,
  "sqft_living": 2200,
  "sqft_lot": 5000,
  "floors": 2.0,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "grade": 8,
  "sqft_above": 1800,
  "sqft_basement": 400,
  "yr_built": 1980,
  "yr_renovated": 0,
  "zipcode": "98115",
  "lat": 47.68,
  "long": -122.30,
  "sqft_living15": 2000,
  "sqft_lot15": 5000
}
```
- **Response Body:**
```json
{
  "prediction": 485000.0,
  "model_version": "1.0",
  "zipcode": "98115",
  "demographic_data_found": true,
  "features_used": 54,
  "timestamp": "2025-09-22T10:30:00Z",
  "metadata": {
    "model_type": "KNeighborsRegressor",
    "confidence_interval": [420000.0, 550000.0],
    "neighbors_considered": 5
  }
}
```

## POST /predict/minimal
- **Purpose:** Predict house price using minimal feature subset (legacy model inputs)
- **Required Fields:** `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `sqft_above`, `sqft_basement`, `zipcode`
- **Behavior:** Missing house features are backfilled with historical medians; demographics added on the server side.
- **Response:** Same schema as `/predict`
