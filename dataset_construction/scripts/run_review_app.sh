#!/usr/bin/env bash
set -euo pipefail

ENV_PY="${ENV_PY:-/opt/anaconda3/envs/openvter-obb-sam/bin/python}"
ENV_STREAMLIT="${ENV_STREAMLIT:-/opt/anaconda3/envs/openvter-obb-sam/bin/streamlit}"
PORT="${PORT:-8501}"
HOST="${HOST:-127.0.0.1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT/logs"
"$ENV_STREAMLIT" run "$ROOT/scripts/review_pseudo_obb_app.py" \
  --server.port "$PORT" \
  --server.address "$HOST" \
  --server.headless true
