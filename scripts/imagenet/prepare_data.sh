#!/bin/bash
# One-time ImageNet-256 setup:
#   1. download the DMF-XL/2+ and SiT-XL/2+REPA checkpoints and FID statistics
#      from https://huggingface.co/kyungmnlee/DMF into $DATA_DIR/checkpoints,
#   2. download ILSVRC/imagenet-1k from the Hugging Face Hub (needs HF_TOKEN and
#      the accepted dataset terms; resumable),
#   3. encode SD-VAE (sd-vae-ft-mse) latents into $LATENT_OUTPUT_DIR (resumable),
#   4. compute the FD-DINOv2 reference statistics into $DINO_REF_STATS.
# STEPS="dmf" runs only step 1 (enough for sampling from released models).
source "$(dirname "$0")/../_env.sh"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$DATA_DIR/imagenet/hf_cache}"
CKPT_DIR="$DATA_DIR/checkpoints"
mkdir -p "$CKPT_DIR" "$HF_DATASETS_CACHE" "$LATENT_OUTPUT_DIR" "$(dirname "$DINO_REF_STATS")"

uv run --frozen hf download kyungmnlee/DMF dmf_xl_2_256.pt sit_xl_2_repa_256.pt \
  imgnet256.pkl --local-dir "$CKPT_DIR"
py -m scripts.imagenet.convert_fid_stats --pkl "$CKPT_DIR/imgnet256.pkl" --out "$FID_REF_STATS"
[[ "${STEPS:-all}" == dmf ]] && exit 0

: "${HF_TOKEN:?Set HF_TOKEN after accepting the ILSVRC/imagenet-1k terms}"
py -c "from datasets import load_dataset; load_dataset('ILSVRC/imagenet-1k', \
cache_dir='$HF_DATASETS_CACHE', token=True)"

HF_DATASETS_OFFLINE=1 py -m scripts.imagenet.encode_latents \
  --cache-dir "$HF_DATASETS_CACHE" --output-dir "$LATENT_OUTPUT_DIR" \
  --split train --shard-size 1024 --batch-size 32

SNAPSHOTS="${HF_HOME:-$HOME/.cache/huggingface}/hub/datasets--ILSVRC--imagenet-1k/snapshots"
py -m scripts.imagenet.compute_dinov2_ref --output "$DINO_REF_STATS" \
  --data-root "${DATA_ROOT:-$SNAPSHOTS}"
