#!/bin/bash
# FKC evaluation behind the paper (after scripts/fkc/train.sh). Writes results/fkc/:
#   table/, diffusion_comparison/   metrics with/without FKC
#   clip_sweep/summary.csv          log-weight clipping sweep
#   tables/*.tex                    appendix clipping tables
#   panels/*.csv                    4-NFE sample panels (25% clipping)
# SEEDS="0" gives a quick single-seed pass (paper: 0 1 2 3).
source "$(dirname "$0")/../_env.sh"
export CUDA_VISIBLE_DEVICES=""

MAP="$RUNS_DIR/fkc40_issfm"
DIFF="$RUNS_DIR/fkc40_diffusion"
OUT=results/fkc
read -r -a SEEDS <<< "${SEEDS:-0 1 2 3}"
BOTH=(--run-dir "$MAP" --diffusion-run-dir "$DIFF")

for seed in "${SEEDS[@]}"; do
  py -m scripts.fkc.metrics_table --run-dir "$MAP" --seed "$seed" --out-dir "$OUT/table"
  py -m scripts.fkc.metrics_table "${BOTH[@]}" --diffusion-only --seed "$seed" \
    --out-dir "$OUT/diffusion_comparison"
  py -m scripts.fkc.metrics_table "${BOTH[@]}" --seed "$seed" --nfe 12 20 24 --k 8 50 \
    --out-dir "$OUT/diffusion_comparison" --file-tag _nfe12_20_24
  py -m scripts.fkc.ess "${BOTH[@]}" --seeds "$seed" --out-dir "$OUT/diffusion_comparison"
  py -m scripts.fkc.ess "${BOTH[@]}" --seeds "$seed" --nfe 12 20 24 --k 1 8 50 \
    --out-dir "$OUT/diffusion_comparison" --file-tag _nfe12_20_24
  py -m scripts.fkc.clip_sweep "${BOTH[@]}" --seed "$seed" --out-dir "$OUT/clip_sweep"
done

py -m scripts.fkc.summarize --run fkc40_issfm --seeds "${SEEDS[@]}" \
  --sweep-dir "$OUT/clip_sweep" --baseline-dir "$OUT/diffusion_comparison" \
  --flow-map-dir "$OUT/table" --out "$OUT/clip_sweep/summary.csv"
py -m scripts.fkc.make_tables --summary "$OUT/clip_sweep/summary.csv" --out-dir "$OUT/tables"
py -m scripts.fkc.panels "${BOTH[@]}" --clip 0.25 --out-dir "$OUT/panels"
echo "FKC results in $OUT/"
