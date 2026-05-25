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
#SBATCH --output=logs/server39/slurm/scene-%j.out
#SBATCH --error=logs/server39/slurm/scene-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch scripts/submit_scene_job.sh <scene_dir> <road_config_dir> [extra run_scene args...]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate server39

bash scripts/run_scene_server.sh "$@"
