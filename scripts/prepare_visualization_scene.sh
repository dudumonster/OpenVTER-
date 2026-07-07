#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/prepare_visualization_scene.sh <scene_output_dir> <scene_name> [--work-root <dir>] [--no-convert] [--no-force]

Example:
  bash scripts/prepare_visualization_scene.sh \
    /public/home/dudu030900/road_config/drone_data_video/dong_guan/ban_xian_shan/output_server39 \
    ban_xian_shan
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

SCENE_OUTPUT_DIR="$1"
SCENE_NAME="$2"
shift 2

WORK_ROOT="/public/home/dudu030900/road_config/visualization/${SCENE_NAME}"
DO_CONVERT=1
FORCE_FLAG="--force"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-root)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --work-root" >&2
        exit 1
      fi
      WORK_ROOT="$2"
      shift 2
      ;;
    --no-convert)
      DO_CONVERT=0
      shift
      ;;
    --no-force)
      FORCE_FLAG=""
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCENE_OUTPUT_DIR="$(realpath "${SCENE_OUTPUT_DIR}")"
WORK_ROOT="$(realpath -m "${WORK_ROOT}")"
INITIAL_ROOT="${WORK_ROOT}/Initial results"
ADJUSTED_ROOT="${WORK_ROOT}/Adjusted results"
FINAL_ROOT="${WORK_ROOT}/Final Data"

if [[ ! -d "${SCENE_OUTPUT_DIR}" ]]; then
  echo "Scene output directory does not exist: ${SCENE_OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${INITIAL_ROOT}" "${ADJUSTED_ROOT}" "${FINAL_ROOT}"

linked_count=0
skipped_count=0
for dir in "${SCENE_OUTPUT_DIR}"/"${SCENE_NAME}"_[0-9][0-9][0-9]; do
  [[ -d "${dir}" ]] || continue
  name="$(basename "${dir}")"
  target="${INITIAL_ROOT}/${name}"

  if [[ -L "${target}" || -f "${target}" ]]; then
    rm -f "${target}"
  elif [[ -e "${target}" ]]; then
    echo "Skip existing non-symlink path: ${target}" >&2
    skipped_count=$((skipped_count + 1))
    continue
  fi

  ln -s "${dir}" "${target}"
  linked_count=$((linked_count + 1))
done

echo "Project root : ${PROJECT_ROOT}"
echo "Scene output : ${SCENE_OUTPUT_DIR}"
echo "Work root    : ${WORK_ROOT}"
echo "Initial root : ${INITIAL_ROOT}"
echo "Adjusted root: ${ADJUSTED_ROOT}"
echo "Final root   : ${FINAL_ROOT}"
echo "Linked dirs  : ${linked_count}"
echo "Skipped dirs : ${skipped_count}"

if [[ "${linked_count}" -eq 0 ]]; then
  echo "No directories matched: ${SCENE_OUTPUT_DIR}/${SCENE_NAME}_[0-9][0-9][0-9]" >&2
  exit 1
fi

if [[ "${DO_CONVERT}" -eq 1 ]]; then
  cd "${PROJECT_ROOT}"
  python3 Visualization/app/converter.py \
    --source-root "${INITIAL_ROOT}" \
    --output-root "${ADJUSTED_ROOT}" \
    --final-output-root "${FINAL_ROOT}" \
    ${FORCE_FLAG}
fi

cat <<EOF

Start visualizer with:
python3 Visualization/app/server.py \\
  --host 127.0.0.1 \\
  --port 8000 \\
  --initial-root "${INITIAL_ROOT}" \\
  --adjusted-root "${ADJUSTED_ROOT}" \\
  --final-root "${FINAL_ROOT}"

Open with SSH tunnel:
ssh -L 8000:127.0.0.1:8000 dudu030900@<server-ip>
http://127.0.0.1:8000
EOF
