#!/usr/bin/env bash
# Submit one OpenVTER scene run through Slurm.
#
# Usage:
#   sbatch scripts/submit_scene_job.sh <scene_dir> <road_config_dir> [extra run_scene args...]
#
# Examples:
#   sbatch scripts/submit_scene_job.sh \
#     /public/home/dudu030900/road_config/drone_data_video/chong_qing/yin_hai_1 \
#     /public/home/dudu030900/road_config/chong_qing/yin_hai_1
#
#   sbatch scripts/submit_scene_job.sh \
#     /public/home/dudu030900/road_config/drone_data_video/chong_qing/yin_hai_1 \
#     /public/home/dudu030900/road_config/chong_qing/yin_hai_1 \
#     --video-range 001-005

#SBATCH --job-name=openvter_scene
#SBATCH --output=logs/scene-%j.out
#SBATCH --error=logs/scene-%j.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=72:00:00

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch scripts/submit_scene_job.sh <scene_dir> <road_config_dir> [extra run_scene args...]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
mkdir -p logs/server39/slurm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate server39

bash scripts/run_scene_server.sh "$@"
