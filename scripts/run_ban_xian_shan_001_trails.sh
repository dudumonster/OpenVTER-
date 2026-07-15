#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_ban_xian_shan_001_trails.sh preview
  bash scripts/run_ban_xian_shan_001_trails.sh full

Environment overrides:
  PROJECT_ROOT, WORK_ROOT, ORIGINAL_VIDEO, PYTHON_BIN, MIN_FREE_GB

Run "preview" first. Inspect the four preview videos and validation frames before
running "full". All outputs are written below WORK_ROOT/trail_videos.
EOF
}

MODE="${1:-preview}"
if [[ "${MODE}" != "preview" && "${MODE}" != "full" ]]; then
  usage
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/dudu030900/Code/OpenVTER}"
WORK_ROOT="${WORK_ROOT:-/public/home/dudu030900/road_config/visualization/ban_xian_shan}"
ORIGINAL_VIDEO="${ORIGINAL_VIDEO:-/public/home/dudu030900/road_config/drone_data_video/dong_guan/ban_xian_shan/ban_xian_shan_001.MP4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"

DATASET_ID="ban_xian_shan_001"
INITIAL_DIR="${WORK_ROOT}/Initial results/${DATASET_ID}"
ADJUSTED_DIR="${WORK_ROOT}/Adjusted results/${DATASET_ID}/moving_filtered"
TRACKING_VIDEO="${INITIAL_DIR}/tracking_output_stab_det_${DATASET_ID}.mp4"
DET_PKL="${INITIAL_DIR}/det_bbox_result_${DATASET_ID}.pkl"
STAB_PKL="${INITIAL_DIR}/${DATASET_ID}_stab.pkl"
TRACKS_CSV="${ADJUSTED_DIR}/${DATASET_ID}_tracks.csv"
TRACKS_META="${ADJUSTED_DIR}/${DATASET_ID}_tracksMeta.csv"
OUTPUT_ROOT="${WORK_ROOT}/trail_videos/${DATASET_ID}"

required_dirs=("${PROJECT_ROOT}" "${WORK_ROOT}")
required_files=(
  "${ORIGINAL_VIDEO}"
  "${TRACKING_VIDEO}"
  "${DET_PKL}"
  "${STAB_PKL}"
  "${TRACKS_CSV}"
  "${TRACKS_META}"
)

for path in "${required_dirs[@]}"; do
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: required directory is not visible inside the container: ${path}" >&2
    exit 3
  fi
done
for path in "${required_files[@]}"; do
  if [[ ! -r "${path}" ]]; then
    echo "ERROR: required input is not readable inside the container: ${path}" >&2
    exit 3
  fi
done
if [[ ! -w "${WORK_ROOT}" ]]; then
  echo "ERROR: persistent visualization root is not writable: ${WORK_ROOT}" >&2
  exit 3
fi

"${PYTHON_BIN}" -c "import cv2, numpy; print('OpenCV', cv2.__version__, 'NumPy', numpy.__version__)"

available_kb="$(df -Pk "${WORK_ROOT}" | awk 'NR==2 {print $4}')"
required_kb="$((MIN_FREE_GB * 1024 * 1024))"
if [[ -z "${available_kb}" || "${available_kb}" -lt "${required_kb}" ]]; then
  echo "ERROR: ${WORK_ROOT} needs at least ${MIN_FREE_GB} GiB free space." >&2
  df -h "${WORK_ROOT}" >&2
  exit 4
fi

mkdir -p \
  "${OUTPUT_ROOT}/previews" \
  "${OUTPUT_ROOT}/validation_frames" \
  "${OUTPUT_ROOT}/reports"

cd "${PROJECT_ROOT}"

PREVIEW_START=2079
PREVIEW_FRAMES=899
FULL_START=0
FULL_FRAMES=9024

if [[ "${MODE}" == "preview" ]]; then
  START_FRAME="${PREVIEW_START}"
  NUM_FRAMES="${PREVIEW_FRAMES}"
  SAMPLE_FRAMES="2079,2528,2977"
  OUTPUT_DIR="${OUTPUT_ROOT}/previews"
  SUFFIX="_preview_2079_899f"
else
  START_FRAME="${FULL_START}"
  NUM_FRAMES="${FULL_FRAMES}"
  SAMPLE_FRAMES="0,2079,2528,2977,9023"
  OUTPUT_DIR="${OUTPUT_ROOT}"
  SUFFIX=""
fi

run_render() {
  local video_source="$1"
  local trail_mode="$2"
  local output_name="$3"
  shift 3
  local output_path="${OUTPUT_DIR}/${output_name}${SUFFIX}.mp4"

  "${PYTHON_BIN}" scripts/overlay_processed_obb_on_video.py \
    --dataset-id "${DATASET_ID}" \
    --visualization-dir "${WORK_ROOT}" \
    --artifact-root "${OUTPUT_ROOT}" \
    --video-source "${video_source}" \
    --trail-mode "${trail_mode}" \
    --trail-seconds 17 \
    --trail-width 4 \
    --max-link-gap-frames 30 \
    --show-legend \
    --start-frame "${START_FRAME}" \
    --num-frames "${NUM_FRAMES}" \
    --frames "${SAMPLE_FRAMES}" \
    --output-video "${output_path}" \
    "$@"

  "${PYTHON_BIN}" - "${output_path}" "${NUM_FRAMES}" <<'PY'
import cv2
import sys

path = sys.argv[1]
expected_frames = int(sys.argv[2])
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    raise SystemExit(f"validation failed: cannot open {path}")
actual = {
    "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    "fps": cap.get(cv2.CAP_PROP_FPS),
    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
}
cap.release()
if actual["frames"] != expected_frames:
    raise SystemExit(f"validation failed: {path}: {actual}")
if (actual["width"], actual["height"]) != (3840, 2160):
    raise SystemExit(f"validation failed: {path}: {actual}")
if abs(actual["fps"] - 29.97) > 0.02:
    raise SystemExit(f"validation failed: {path}: {actual}")
print("validated", path, actual)
PY
}

run_render \
  original finite \
  "${DATASET_ID}_original_moving_filtered_trail_17s_noid" \
  --video-path "${ORIGINAL_VIDEO}" \
  --version moving_filtered \
  --draw-space original \
  --draw-boxes \
  --hide-labels

run_render \
  original permanent \
  "${DATASET_ID}_original_moving_filtered_trail_permanent_noid" \
  --video-path "${ORIGINAL_VIDEO}" \
  --version moving_filtered \
  --draw-space original \
  --draw-boxes \
  --hide-labels

run_render \
  tracking finite \
  "${DATASET_ID}_tracking_stab_det_trail_17s" \
  --video-path "${TRACKING_VIDEO}" \
  --no-draw-boxes \
  --hide-labels

run_render \
  tracking permanent \
  "${DATASET_ID}_tracking_stab_det_trail_permanent" \
  --video-path "${TRACKING_VIDEO}" \
  --no-draw-boxes \
  --hide-labels

echo "Completed ${MODE} renders under: ${OUTPUT_ROOT}"
