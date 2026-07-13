#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

download() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  echo "Downloading: $out"
  curl -L -C - --fail --retry 5 --connect-timeout 30 -o "$out" "$url"
}

case "${1:-help}" in
  visdrone)
    download "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip" "$ROOT/data_sources/visdrone/downloads/VisDrone2019-DET-train.zip"
    download "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip" "$ROOT/data_sources/visdrone/downloads/VisDrone2019-DET-val.zip"
    download "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip" "$ROOT/data_sources/visdrone/downloads/VisDrone2019-DET-test-dev.zip"
    ;;
  uav-obb)
    download "https://data.mendeley.com/public-files/datasets/6snrjwcpkh/files/b1440f9d-0d71-40a9-bc9a-76481d02de7e/file_downloaded" "$ROOT/data_sources/uav_obb/downloads/UAV-OBB-dlaCi7.zip"
    ;;
  vsai)
    echo "VSAI is about 13.9GB. This Supervisely mirror can be very slow."
    echo "If it is too slow, stop with Ctrl+C and use the Kaggle option described in download_manifest.md."
    download "https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMTMzNl9WU0FJL3ZzYWktRGF0YXNldE5pbmphLnRhciIsICJzaWciOiAid3NQQ2ZVTVRlR3JBZm5heXVrK3kvaHEra0h3MVZTSzZzell4RWg2UzBrVT0ifQ==?response-content-disposition=attachment%3B%20filename%3D%22vsai-DatasetNinja.tar%22" "$ROOT/data_sources/vsai/downloads/vsai-DatasetNinja.tar"
    ;;
  all-direct)
    "$0" uav-obb
    "$0" visdrone
    ;;
  all-with-vsai)
    "$0" uav-obb
    "$0" visdrone
    "$0" vsai
    ;;
  help|*)
    cat <<'EOF'
Usage:
  bash dataset_construction/scripts/download_datasets.sh uav-obb
  bash dataset_construction/scripts/download_datasets.sh visdrone
  bash dataset_construction/scripts/download_datasets.sh vsai
  bash dataset_construction/scripts/download_datasets.sh all-direct
  bash dataset_construction/scripts/download_datasets.sh all-with-vsai

Notes:
  - Downloads use curl -C -, so interrupted files resume from the existing partial file.
  - VisDrone is HBB and is mainly used for weak traffic participant classes.
  - all-direct intentionally excludes VSAI because VSAI is large and the mirror can be slow.
  - Use all-with-vsai only when you really want to fetch the large VSAI tar.
  - DroneVehicle, DOTA, and EAGLE require their official pages or Google/Baidu-style download flows.
EOF
    ;;
esac
