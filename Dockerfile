FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY data data
COPY model model
COPY scripts scripts
COPY .env.example .env.example

RUN adduser --disabled-password --gecos '' appuser
USER appuser

EXPOSE 8000

ENV MODEL_DIR=/app/model \
    MODEL_FILENAME=model.pkl \
    MODEL_FEATURES_FILE=model_features.json \
    MODEL_INFO_FILE=model_info.json \
    DEMOGRAPHICS_PATH=/app/data/zipcode_demographics.csv \
    SALES_DATA_PATH=/app/data/kc_house_data.csv

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD python -c "import sys, urllib.request;\nreq = urllib.request.Request('http://127.0.0.1:8000/health');\nreq.add_header('Connection','close');\nurllib.request.urlopen(req, timeout=5)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
