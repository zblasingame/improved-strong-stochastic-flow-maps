---
license: mit
library_name: pytorch
pipeline_tag: unconditional-image-generation
tags:
  - flow-maps
  - stochastic-differential-equations
  - imagenet
datasets:
  - ILSVRC/imagenet-1k
base_model: kyungmnlee/DMF
---

# Improved Training of Strong Stochastic Flow Maps (iSSFM)

[![arXiv](https://img.shields.io/badge/arXiv-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2610.29605)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github)](https://github.com/zblasingame/improved-strong-stochastic-flow-maps)

The model checkpoints for (improved) strong stochastic flow maps for class-conditional ImageNet-256 generation, fine-tuned
from [DMF-XL/2+](https://huggingface.co/kyungmnlee/DMF).
A strong stochastic flow map jumps along the solution of an SDE for a *given* Brownian path, so samples driven
by the same path converge to the same image as the number of steps (NFE) grows.

## Models

We provide our reproduction of SSFM in this setting along with our improved parameterization and losses (see Tables 1 and 2); each model adds one change to the previous one.
These are all fine-tuned from DMF-XL/2+ for 100k steps with $N=3$ shifted Legendre coefficients.

| Folder | Description | FID-50k @ 4 NFE | FD<sub>DINOv2</sub> @ 4 NFE |
| --- | --- | --- | --- |
| `ssfm` | SSFM | 31.00 | 475.1 |
| `ssfm_lift` | + exponential lift | 28.15 | 447.4 |
| `ssfm_lift_martingale` | + closed-form martingale | 18.33 | 301.6 |
| `ssfm_lift_martingale_drift` | + drift matching | 14.67 | 288.1 |
| `issfm` | iSSFM (+ Wong--Zakai iMF loss) | **5.96** | **143.0** |

Each folder holds `model.safetensors` (EMA weights of the fine-tuned parameters:
LoRA adapters, Brownian-coefficient injection and, for `ssfm` / `ssfm_lift`, the
learned diffusion head; 150–165 MB) and `config.yaml`.
The frozen DMF-XL/2+ backbone is **not** included; download `dmf_xl_2_256.pt` from
[kyungmnlee/DMF](https://huggingface.co/kyungmnlee/DMF).

## Usage

```bash
git clone https://github.com/zblasingame/improved-strong-stochastic-flow-maps && cd improved-strong-stochastic-flow-maps
uv sync
hf download kyungmnlee/DMF dmf_xl_2_256.pt --local-dir checkpoints
export DMF_CHECKPOINT_PATH=checkpoints/dmf_xl_2_256.pt
uv run python -m scripts.imagenet.sample --model issfm --labels 339 207 --nfe 1 2 4 8 16
```

This writes `samples.png`: one row per class, with every NFE driven by the same
initial noise and Brownian path, à la Figure 5 in the paper.
For more details see the [GitHub repository](https://github.com/zblasingame/improved-strong-stochastic-flow-maps).

## Attribution

Built on DMF-XL/2+ ([Decoupled MeanFlow](https://github.com/kyungmnlee/dmf)), itself fine-tuned from SiT-XL/2+REPA.

## Citation

Please cite the following paper if you used these models in a research context.

```bibtex
@misc{blasingame2026improved,
  title={Improved Training of Strong Stochastic Flow Maps},
  author={Blasingame, Zander W. and Franca, Antonio and Mi{\~n}{\'a}n, Ra{\'u}l and Koke, Christian and Rindtorff, Niklas and Tong, Alexander},
  year={2026},
  eprint={2610.29605},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2610.29605}
}
```
