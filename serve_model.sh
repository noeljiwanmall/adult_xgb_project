#!/usr/bin/env bash
set -euo pipefail

echo "⏳ waiting for best_run.txt ..."
while [ ! -s /app/best_run.txt ]; do sleep 1; done

RUN_ID=$(tr -d '\r\n' < /app/best_run.txt)
echo "📋 best run id: $RUN_ID"

export MLFLOW_TRACKING_URI="file:/app/mlruns"

# ── wait until the pipeline_model directory actually exists ─────────────
echo "⏳ waiting for pipeline_model artifacts ..."
until find /app/mlruns -type f -path "*/${RUN_ID}/artifacts/pipeline_model/MLmodel" -print -quit 2>/dev/null; do
  sleep 2
done
echo "✅ artifacts ready – starting server"

# Listen on all interfaces so other containers / host can connect
exec mlflow models serve \
     -m "runs:/${RUN_ID}/pipeline_model" \
     --host 0.0.0.0 --port 5002 --no-conda
