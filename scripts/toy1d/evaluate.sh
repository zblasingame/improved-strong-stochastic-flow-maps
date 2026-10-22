#!/bin/bash
# Strong error of the 36 toy-1D runs on one shared realization (1024 initial
# conditions and Brownian paths, 512-step Euler-Maruyama reference, seed 12345)
# for NFE in {1,...,32}. Writes results/toy1d/strong_error.{png,csv}.
source "$(dirname "$0")/../_env.sh"

RUN_ARGS=()
for run in "$RUNS_DIR"/toy1d_*_N[1-4]; do
  RUN_ARGS+=(--run-dir "$run")
done
mkdir -p results/toy1d
py -m scripts.toy1d.evaluate "${RUN_ARGS[@]}" \
  --num-samples 1024 --em-steps 512 --seed 12345 --steps-plot 16 \
  --out results/toy1d/strong_error.png
