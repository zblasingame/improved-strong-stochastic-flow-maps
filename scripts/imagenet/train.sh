#!/bin/bash
# Fine-tune DMF-XL/2+ into a strong stochastic flow map on ImageNet-256 latents.
#   scripts/imagenet/train.sh EXPERIMENT [hydra overrides...]
# EXPERIMENT (configs/experiment/imagenet256_<EXPERIMENT>.yaml):
#   Ablation ladder (paper Table 1), each rung adding to the previous one:
#     ssfm                          SSFM (learned noise, no lift, EM-step diagonal)
#     ssfm_lift                     + exponential lift
#     ssfm_lift_martingale          + closed-form martingale
#     ssfm_lift_martingale_drift    + drift matching (improved parameterization)
#     wz_imf                        iSSFM: + Wong--Zakai iMF loss
# Examples:
#   scripts/imagenet/train.sh wz_imf
#   RUN_NAME=wz_imf_n6 scripts/imagenet/train.sh wz_imf model.net.degree=6 \
#     model.loss.degree=6 callbacks.sample_grid.degree=6 callbacks.fid.degree=6
# Resume with CKPT_PATH=/path/to/current.ckpt.
source "$(dirname "$0")/../_env.sh"

EXPERIMENT="${1:?usage: train.sh ssfm|ssfm_lift|ssfm_lift_martingale|ssfm_lift_martingale_drift|wz_imf [overrides...]}"
shift
require_file "$LATENT_OUTPUT_DIR/train/manifest.json" "latents (run scripts/imagenet/prepare_data.sh)"
require_file "$DMF_CHECKPOINT_PATH" "DMF-XL/2 checkpoint (DMF_CHECKPOINT_PATH)"

train "imagenet256_$EXPERIMENT" "imagenet256_${RUN_NAME:-$EXPERIMENT}" "$@"
