# Morri Solution Overview

This report documents the work completed for the Sound Realty price prediction service, covering model deployment, model selection, testing, and supporting artifacts.

## Highlights

- **FastAPI service** (see `app/`): serves `/predict`, `/predict/minimal`, `/model/info`, and `/health` endpoints with automatic demographic enrichment and singleton model loading.
- **Model lifecycle tooling** (`scripts/`):
  - `train_model.py` loads sales + demographic data, tunes four algorithm families (kNN, RandomForest, GradientBoosting, HistGradientBoosting) via GridSearchCV, and exports the champion model.
  - `evaluate_model.py` re-evaluates the selected model on the holdout split (same data split as training) and writes summary metrics to `model/model_evaluation.json`.
  - `test_api.py` exercises both prediction endpoints using examples from `data/future_unseen_examples.csv`.
  - `plot_model_comparison.py` generates comparison charts from `model/model_info.json` and writes them to `reports/`.
- **Artifacts** (`model/`):
  - `model.pkl`, `model_features.json`, `model_info.json` (includes full leaderboard) and `model_evaluation.json`.
- **Testing** (`tests/`): pytest suite covering API responses and prediction service logic; `uv run python -m pytest` is green.
- **Documentation**:
  - `reports/model_comparison.md` summarises the model bake-off and includes reproduction steps.
  - `reports/*.png` contain validation MAE, test R², and champion-vs-baseline plots.
  - `README.md` updated with plotting instructions.

## Results Summary (latest run)

Source: `model/model_info.json` and `model/model_evaluation.json`

- **Champion model**: GradientBoostingRegressor (`learning_rate=0.1`, `max_depth=5`, `n_estimators=400`).
- **Validation MAE**: $67,280 (down ~20% from tuned kNN baseline).
- **Test R²**: 0.887 (up from 0.793 baseline).
- **RMSE on evaluation split**: $130,959 (`model/model_evaluation.json`).
- **Candidate comparison**:
  - HistGradientBoostingRegressor: MAE $68,189, R² 0.887.
  - RandomForestRegressor: MAE $70,612, R² 0.877.
  - KNeighborsRegressor: MAE $84,494, R² 0.793.

## Reproduction Checklist

From project root:

1. Install dependencies: `pip install -r requirements.txt` (includes matplotlib).
2. Train and compare models: `UV_CACHE_DIR=.uv-cache uv run python scripts/train_model.py`.
3. Evaluate champion: `UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_model.py`.
4. (Optional) Generate plots: `UV_CACHE_DIR=.uv-cache uv run python scripts/plot_model_comparison.py`.
5. Run tests: `uv run python -m pytest`.
6. Review artifacts: `model/`, `reports/`, and `tests/` outputs per the sections above.

## Next Steps (optional ideas)

- Automate training + evaluation in CI/CD and publish plots/metrics per commit.
- Add observability hooks (e.g., structured logging, latency metrics) to support production deployments.
- Explore interpretability (SHAP, PDP) for stakeholder reports.

