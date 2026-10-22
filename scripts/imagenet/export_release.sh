#!/bin/bash
# Export trained runs into release/<name>/{model.safetensors,config.yaml}.
# Pass name=run_dir[:checkpoint] (run_dir relative to $RUNS_DIR or absolute;
# checkpoint defaults to CKPT_NAME, else current.ckpt). The released models are
# the 100k-step runs of the paper's ablation ladder (Table 1) and iSSFM:
#   scripts/imagenet/export_release.sh \
#     ssfm=imagenet256_ssfm ssfm_lift=imagenet256_ssfm_lift \
#     ssfm_lift_martingale=imagenet256_ssfm_lift_martingale \
#     ssfm_lift_martingale_drift=imagenet256_ssfm_lift_martingale_drift \
#     issfm=imagenet256_wz_imf
# Upload with: hf upload zblasingame/improved-strong-stochastic-flow-maps release/ .
source "$(dirname "$0")/../_env.sh"

for pair in "$@"; do
  name="${pair%%=*}" spec="${pair#*=}"
  run="${spec%%:*}" ckpt="${CKPT_NAME:-current.ckpt}"
  [[ "$spec" == *:* ]] && ckpt="${spec#*:}"
  [[ "$run" == /* ]] || run="$RUNS_DIR/$run"
  py -m scripts.imagenet.export_checkpoint --run-dir "$run" \
    --checkpoint "$ckpt" --out-dir "release/$name"
done
