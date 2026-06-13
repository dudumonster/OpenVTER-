#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_scene_server.sh <scene_dir> <road_config_dir> [extra args...]"
  exit 1
fi

SCENE_DIR="$1"
ROAD_CONFIG_DIR="$2"
shift 2
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

CPU_THREADS="${SLURM_CPUS_PER_TASK:-${OPENVTER_CPU_THREADS:-8}}"
export OPENVTER_CPU_THREADS="${CPU_THREADS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_THREADS}}"
export PYTHONUNBUFFERED=1
export TORCH_CUDNN_V8_API_DISABLED="${TORCH_CUDNN_V8_API_DISABLED:-1}"

mkdir -p logs/server39/slurm

FORCE=false
VIDEO_NAME=""
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  case "${EXTRA_ARGS[$i]}" in
    --force)
      FORCE=true
      ;;
    --video)
      if (( i + 1 >= ${#EXTRA_ARGS[@]} )); then
        echo "ERROR: --video requires a value." >&2
        exit 2
      fi
      VIDEO_NAME="${EXTRA_ARGS[$((i + 1))]}"
      ;;
    --video=*)
      VIDEO_NAME="${EXTRA_ARGS[$i]#--video=}"
      ;;
  esac
done

if [[ "${FORCE}" == "true" && -n "${VIDEO_NAME}" ]]; then
  VIDEO_STEM="$(basename "${VIDEO_NAME}")"
  VIDEO_STEM="${VIDEO_STEM%.*}"
  OLD_OUTPUT_DIR="${SCENE_DIR%/}/output_server39/${VIDEO_STEM}"
  if [[ -d "${OLD_OUTPUT_DIR}" ]]; then
    BACKUP_DIR="${OLD_OUTPUT_DIR}.bak_$(date +%Y%m%d_%H%M%S)"
    echo "Backing up old output: ${OLD_OUTPUT_DIR} -> ${BACKUP_DIR}"
    mv "${OLD_OUTPUT_DIR}" "${BACKUP_DIR}"
  fi
fi

python3 -u src/run_scene.py \
  --config configs/server.yaml \
  --scene-dir "${SCENE_DIR}" \
  --road-config-dir "${ROAD_CONFIG_DIR}" \
  "${EXTRA_ARGS[@]}"
