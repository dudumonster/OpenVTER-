#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_scene_server.sh <scene_dir> <road_config_dir> [extra args...]"
  exit 1
fi

SCENE_DIR="$1"
ROAD_CONFIG_DIR="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
python3 src/run_scene.py \
  --config configs/server.yaml \
  --scene-dir "${SCENE_DIR}" \
  --road-config-dir "${ROAD_CONFIG_DIR}" \
  "$@"
