# Sound Realty Price Prediction Service

Production-ready FastAPI service that serves a kNN house price model for the phData MLE project. The API enriches incoming requests with census demographics, supports minimal legacy feature payloads, and is packaged with Docker, automated evaluation, and integration tests.

## Quick Start
1. Create virtual environment (Python 3.9):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
pip install -r requirements.txt
```
2. Train the model and export artifacts:
   ```bash
   python scripts/train_model.py
   ```
3. (Optional) Evaluate performance:
   ```bash
python scripts/evaluate_model.py
```
6. Generate comparison plots (optional):
   ```bash
   python scripts/plot_model_comparison.py
   ```
4. Launch the API locally:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
5. Exercise endpoints:
   ```bash
   python scripts/test_api.py --limit 3
   ```

## Project Layout
```
app/                # FastAPI application package
  data_loader.py    # Demographics caching utilities
  main.py           # API routes and FastAPI app definition
  models.py         # Pydantic request/response schemas
  predictor.py      # Singleton model service
  utils.py          # Shared helpers and settings
scripts/            # Operational scripts
  train_model.py    # Training entry point
  evaluate_model.py # Offline evaluation metrics
  test_api.py       # Sample client exercising endpoints
tests/              # Pytest suite for predictor and API
Dockerfile          # Production image definition
docker-compose.yml  # Local orchestration config
requirements.txt    # Python dependencies
.env.example        # Environment template for runtime configuration
api_docs.md         # Endpoint reference
```

## Configuration
Copy `.env.example` to `.env` to override defaults. Key variables:
- `MODEL_DIR`, `MODEL_FILENAME`, `MODEL_FEATURES_FILE`, `MODEL_INFO_FILE`
- `DEMOGRAPHICS_PATH`, `SALES_DATA_PATH`
- `API_TITLE`, `API_VERSION`, `API_DESCRIPTION`

## Endpoints
- `GET /health` – Liveness probe
- `GET /model/info` – Model metadata (`model_info.json`)
- `POST /predict` – Full feature payload (matches `future_unseen_examples.csv` columns)
- `POST /predict/minimal` – Minimal subset for legacy integrations

Detailed examples live in `api_docs.md`. FastAPI also exposes interactive docs at `/docs` when the service is running.

## Testing
Run unit and integration tests with:
```bash
pytest tests
```

## Continuous Integration
Pushes and pull requests trigger `.github/workflows/ci.yml`, which provisions Python 3.9, installs dependencies, runs model training, records evaluation metrics, executes the pytest suite, and uploads the generated artifacts for inspection.

## Docker Workflow
Build and run with Docker Compose (requires Docker Desktop):
```bash
docker-compose up --build
```
The container uses an unprivileged user, exposes port 8000, and mounts `./model` for hot-swapping trained artifacts.

## Operational Notes
- Demographics data is cached in memory and median-filled if a zipcode lookup fails.
- Confidence intervals are approximated using neighbor predictions from the kNN model to provide directional guidance.
- Minimal prediction requests backfill missing house features using historical medians before enrichment.
- `scripts/evaluate_model.py` writes `model/model_evaluation.json` alongside console output for reporting.

## Next Steps
- Automate artifact refresh via CI/CD, wiring `scripts/train_model.py` + `scripts/evaluate_model.py` in the pipeline.
- Add monitoring hooks (latency, error rates, request volume) and structured log shipping.
- Introduce feature drift detection using rolling window statistics on incoming payloads.
