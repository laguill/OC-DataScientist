#!/usr/bin/env bash
uv run mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///data/mlflow/db/mlflow.db \
  --default-artifact-root ./data/mlflow/artifacts