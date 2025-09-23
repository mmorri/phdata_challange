# Model Comparison Summary

Generated from `model/model_info.json` on the latest training run.

## Leaderboard (Validation Holdout)

| Model | Validation MAE ($) | Test R² | CV MAE ($) | Key Hyperparameters |
| --- | ---: | ---: | ---: | --- |
| GradientBoostingRegressor *(selected)* | 67,279 | 0.887 | 63,998 | `learning_rate=0.1`, `max_depth=5`, `n_estimators=400` |
| HistGradientBoostingRegressor | 68,189 | 0.887 | 65,516 | `learning_rate=0.1`, `max_leaf_nodes=63`, `max_depth=None` |
| RandomForestRegressor | 70,612 | 0.877 | 67,292 | `n_estimators=400`, `max_depth=40`, `min_samples_leaf=1` |
| KNeighborsRegressor | 84,494 | 0.793 | 79,976 | `n_neighbors=7`, `p=1`, `weights=distance` |

## Champion Lift vs. Tuned KNN Baseline

- Absolute MAE improvement: **$17,214** lower.
- Relative MAE improvement: **20.4%** reduction on the validation split.
- Test R² gain: **+0.093** (0.793 → 0.887).
- RMSE from `scripts/evaluate_model.py`: **$130,959** for the champion vs. **$176,980** (previous tuned KNN).

## Reproduction

1. `UV_CACHE_DIR=.uv-cache uv run python scripts/train_model.py`
2. `UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_model.py`
3. `uv run python -m pytest`
4. (Optional) Generate plots: `UV_CACHE_DIR=.uv-cache uv run python scripts/plot_model_comparison.py`
5. Inspect updated metrics in `model/model_info.json`, regenerated figures (`reports/*.png`), and this report.
