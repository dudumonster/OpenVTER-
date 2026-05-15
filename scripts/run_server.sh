#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/server39"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/run_${TIMESTAMP}.log"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"
python3 src/train.py --config configs/server.yaml 2>&1 | tee "${LOG_FILE}"
