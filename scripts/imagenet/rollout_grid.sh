#!/bin/bash
# Strong-solution rollout grid (paper figure "Strong solutions on ImageNet-256"):
# each row is one (X_0, Brownian path) realization at NFE 4..16, keeping the rows
# with the lowest strong error against the 16-NFE rollout.
#   scripts/imagenet/rollout_grid.sh RUN_DIR [CKPT_NAME=current.ckpt]
source "$(dirname "$0")/../_env.sh"

RUN_DIR="${1:?usage: rollout_grid.sh RUN_DIR [CKPT_NAME]}"
CKPT="$RUN_DIR/checkpoints/${2:-current.ckpt}"
require_file "$CKPT" checkpoint
py -m scripts.imagenet.rollout_grid --checkpoint "$CKPT" \
  --config "$RUN_DIR/.hydra/config.yaml" --output-dir results/imagenet/grids \
  --mode strong-select --label-id "${LABEL_ID:-339}" --nfes 4 6 8 10 12 14 16
