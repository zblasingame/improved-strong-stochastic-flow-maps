#!/bin/bash
# 50k-sample ImageNet-256 evaluation: FID, FD-DINOv2, ImageReward, PickScore,
# HPSv2.1 at NFE 1 2 4 8 16. Writes EVAL_DIR/summary.json.
#   scripts/imagenet/evaluate.sh student RUN_DIR [CKPT_NAME=current.ckpt]  # paper: step 100k
#   scripts/imagenet/evaluate.sh release issfm      # released model (Hugging Face)
#   scripts/imagenet/evaluate.sh teacher            # SiT-XL/2+REPA, EM SDE
# NUM_TASKS=<num GPUs> splits generation across GPUs.
source "$(dirname "$0")/../_env.sh"

KIND="${1:?usage: evaluate.sh student RUN_DIR [CKPT] | release MODEL | teacher}"
EVAL_ROOT="${EVAL_ROOT:-$SCRATCH_DIR/eval_samples/imagenet256}"
NUM_TASKS="${NUM_TASKS:-1}"
if [[ "$KIND" == student ]]; then
  RUN_DIR="${2:?student needs RUN_DIR}"
  CKPT="$RUN_DIR/checkpoints/${3:-current.ckpt}"
  require_file "$CKPT" checkpoint
  EVAL_DIR="$EVAL_ROOT/$(basename "$RUN_DIR")/$(basename "$CKPT" .ckpt)"
  GEN_ARGS=(student --checkpoint "$CKPT" --config "$RUN_DIR/.hydra/config.yaml")
elif [[ "$KIND" == release ]]; then
  MODEL="${2:?release needs a model name, e.g. issfm}"
  EVAL_DIR="$EVAL_ROOT/release_$(basename "$MODEL")"
  GEN_ARGS=(release --model "$MODEL")
else
  require_file "$SIT_CHECKPOINT_PATH" "SiT-XL/2+REPA checkpoint (SIT_CHECKPOINT_PATH)"
  EVAL_DIR="$EVAL_ROOT/teacher_sit_xl2_repa"
  GEN_ARGS=(teacher --checkpoint "$SIT_CHECKPOINT_PATH")
fi

for ((i = 0; i < NUM_TASKS; i++)); do
  CUDA_VISIBLE_DEVICES="$i" py -m scripts.imagenet.generate "${GEN_ARGS[@]}" \
    --eval-dir "$EVAL_DIR" --task-index "$i" --num-tasks "$NUM_TASKS" &
done
wait

SCORING_ENV="${SCORING_ENV:-$PROJECT_DIR/scripts/imagenet/scoring_env/.venv}"
UV_PROJECT_ENVIRONMENT="$SCORING_ENV" uv sync --frozen --python 3.11 \
  --project scripts/imagenet/scoring_env
"$SCORING_ENV/bin/python" -m scripts.imagenet.score --eval-dir "$EVAL_DIR" \
  --fid-ref "$FID_REF_STATS" --dino-ref "$DINO_REF_STATS"
py -m scripts.imagenet.write_summary --eval-dir "$EVAL_DIR"
echo "Summary: $EVAL_DIR/summary.json"
