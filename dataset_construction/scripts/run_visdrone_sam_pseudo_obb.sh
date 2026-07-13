#!/usr/bin/env bash
set -euo pipefail

ENV_PY="${ENV_PY:-/opt/anaconda3/envs/openvter-obb-sam/bin/python}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam/sam_vit_b_01ec64.pth}"
SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_b}"
DEVICE="${DEVICE:-mps}"
SPLITS="${SPLITS:-train val}"
COPY_MODE="${COPY_MODE:-symlink}"

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"

"$ENV_PY" dataset_construction/scripts/visdrone_hbb_to_pseudo_obb.py generate \
  --splits $SPLITS \
  --segmenter sam \
  --sam-checkpoint "$SAM_CHECKPOINT" \
  --sam-model-type "$SAM_MODEL_TYPE" \
  --device "$DEVICE" \
  --copy-mode "$COPY_MODE" \
  "$@"
