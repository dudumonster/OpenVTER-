#!/usr/bin/env bash
# If your cluster is not using Slurm, replace this script with the
# corresponding scheduler syntax (PBS, LSF, SGE, or a plain shell script).

#SBATCH --job-name=openvter_train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
mkdir -p logs

# Replace this block with your actual environment activation command.
# Example:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate openvter

python3 src/train.py --config configs/server.yaml
