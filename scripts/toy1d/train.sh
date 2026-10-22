#!/bin/bash
# Toy 1D non-linear SDE: 9 losses x N in {1,2,3,4} = 36 runs (5k steps each).
# TASK_ID=k (0-35) runs one task, e.g. as a job-array element; otherwise all.
# TRAINER=cpu runs without a GPU. Then run scripts/toy1d/evaluate.sh.
source "$(dirname "$0")/../_env.sh"

LOSSES=(wz-esd-ayf_emd wz-esd-imf wz-lsd-ayf_lmd
        ds-esd-ayf_emd ds-esd-imf ds-lsd-ayf_lmd
        none-psd none-discrete_esd none-discrete_lsd)
TASK_ID="${TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}"
TASKS=(${TASK_ID:-$(seq 0 35)})

for task in "${TASKS[@]}"; do
  IFS=- read -r ARM KIND FORM <<< "${LOSSES[$((task / 4))]}"
  N=$((task % 4 + 1))
  train toy1d "toy1d_${ARM}-${KIND}${FORM:+-$FORM}_N${N}" \
    trainer="${TRAINER:-gpu}" model.loss.arm="$ARM" model.loss.distillation="$KIND" \
    ${FORM:+model.loss.form=$FORM} model.loss.degree="$N" model.net.max_degree="$N" "$@"
done
