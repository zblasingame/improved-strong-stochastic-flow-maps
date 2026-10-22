#!/bin/bash
# Shared environment for the train/evaluate scripts (sourced, not executed).
#
# Paths come from the environment or a ``.env`` file in the repository root
# (see ``.env.example``). Scripts can be run from any directory; on a cluster,
# wrap them in your scheduler's submission command.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT="$PROJECT_DIR"
# Load .env without overriding variables already set in the environment.
if [[ -f "$PROJECT_DIR/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [[ -n "${!key+x}" ]] && continue
        value="${value%\"}"; value="${value#\"}"
        export "$key=$value"
    done < <(grep -v -E '^\s*(#|$)' "$PROJECT_DIR/.env")
fi

export SCRATCH_DIR="${SCRATCH_DIR:-$PROJECT_DIR}"
export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
export RUNS_DIR="${RUNS_DIR:-$SCRATCH_DIR/logs/train/runs}"
export LATENT_OUTPUT_DIR="${LATENT_OUTPUT_DIR:-$DATA_DIR/imagenet256-sd-vae-ft-mse}"
export DMF_CHECKPOINT_PATH="${DMF_CHECKPOINT_PATH:-$DATA_DIR/checkpoints/dmf_xl_2_256.pt}"
export SIT_CHECKPOINT_PATH="${SIT_CHECKPOINT_PATH:-$DATA_DIR/checkpoints/sit_xl_2_repa_256.pt}"
export FID_REF_STATS="${FID_REF_STATS:-$DATA_DIR/checkpoints/imgnet256_torchmetrics.pt}"
export DINO_REF_STATS="${DINO_REF_STATS:-$DATA_DIR/eval_refs/imgnet256_dinov2_vitl14.pt}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Logging backend for training: wandb (default) or csv.
export LOGGER="${LOGGER:-wandb}"

cd "$PROJECT_DIR"
py() { uv run --frozen python "$@"; }
require_file() { [[ -f "$1" ]] || { echo "Missing $2: $1" >&2; exit 1; }; }

# Train one Hydra experiment into $RUNS_DIR/<name>; extra args go to Hydra.
train() {
    local experiment="$1" name="$2"; shift 2
    local logger_args=("logger=$LOGGER")
    [[ "$LOGGER" == wandb ]] && logger_args+=("+logger.wandb.name=$name")
    py -m src.train "experiment=$experiment" "${logger_args[@]}" \
        "hydra.run.dir=$RUNS_DIR/$name" ${CKPT_PATH:+"ckpt_path=$CKPT_PATH"} "$@"
}
