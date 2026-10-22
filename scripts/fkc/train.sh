#!/bin/bash
# Feynman-Kac correctors on the 40-Gaussian mixture (CPU is sufficient):
#   1. the augmented iSSFM map on the annealed beta=3 target (WZ-iMF objective),
#   2. a diffusion (score) baseline on the untempered beta=1 target.
# Then run scripts/fkc/evaluate.sh.
source "$(dirname "$0")/../_env.sh"
export CUDA_VISIBLE_DEVICES=""

train fkc_gaussian40 fkc40_issfm trainer=cpu "$@"
train fkc_diffusion fkc40_diffusion trainer=cpu "$@"
