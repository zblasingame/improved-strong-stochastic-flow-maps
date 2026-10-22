r"""Sample fixed-seed image grids from an SSFM Lightning checkpoint.

Two modes:

``independent``: a square grid where every image draws its own Brownian path
per step (the stock sampler — current training-grid behavior).

``chen``: the Chen-coupled NFE comparison grid. Each *row* is an independent
``(X_0, leaf-path)`` realization; each *column* re-integrates that row at a
different NFE, left (1) to right (max). Rows are fresh candidates for
paper-figure selection; ``--pages`` iterates over more realizations, one PNG
per page. Coarse steps see the exact ``concat_arbitrary`` composition of the
same fresh degree-N leaves the finest NFE resolves (the per-step path
distribution the network was trained on), so within a row moving left to
right changes *only* the time discretization — the proper Chen relations,
unlike restricting a single global polynomial (whose leaf features are far
off-distribution and sample badly).

Example (horse class 339)::

    uv run python -m scripts.imagenet.rollout_grid \\
        --checkpoint /path/to/run/checkpoints/last.ckpt \\
        --config /path/to/run/.hydra/config.yaml \\
        --mode chen --label-id 339 --nfes 1 2 4 8 16 --rows 8 --pages 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
import rootutils
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torchvision.utils import save_image

rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

from src.evaluation.checkpoints import load_student  # noqa: E402
from src.models.components.brownian import BrownianPolynomial  # noqa: E402
from src.models.components.ei_doss import ei_doss_map, ei_doss_reconstruct  # noqa: E402
from src.models.components.gaussian import randn  # noqa: E402
from src.models.components.samplers import (  # noqa: E402
    chen_matched_ei_doss_rows,
    ei_doss_sussmann_sampler,
    ei_euler_sampler,
)
from src.models.components.strong_convergence import (  # noqa: E402
    PiecewiseBrownianPath,
    martingale_interval_path,
    sample_piecewise_ei_clock,
)
from src.models.components.vae_decoder import StabilityVAEDecoder  # noqa: E402
from src.models.ei_doss_module import EIDossSussmannLitModule  # noqa: E402

LATENT_SHAPE = (4, 32, 32)


def _bf16_inference(device: torch.device):
    """Use bf16 autocast for network inference on CUDA, preserving CPU behavior."""
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()


def chen_row_samples(
    net: torch.nn.Module,
    schedule,
    latents: Tensor,
    nfes: list[int],
    degree: int,
    t0: float,
    zeta: float,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    """Thin wrapper over :func:`chen_matched_ei_doss_rows` (kept for the CLI).

    Integrate one shared ``X_0`` under per-row leaf-composed paths at every
    NFE: fresh degree-``degree`` leaves on the finest clock grid, composed
    with ``BrownianPolynomial.concat_arbitrary`` (exact Chen relations) for
    every coarser NFE. Returns ``(len(nfes), num_paths, *LATENT_SHAPE)``.
    """
    del device
    return chen_matched_ei_doss_rows(
        net,
        schedule,
        latents,
        nfes,
        degree=degree,
        t0=t0,
        zeta=zeta,
        generator=generator,
    )


def ei_doss_rollout(
    net: torch.nn.Module,
    schedule,
    latents: Tensor,
    realization: PiecewiseBrownianPath,
    nfe: int,
    degree: int,
    t0: float,
    zeta: float,
) -> Tensor:
    """EI-Doss rollout at one NFE on a shared piecewise EI-clock realization.

    Mirrors ``ei_doss_rollout_batch``'s coarse rollout: each step's martingale
    view comes from ``martingale_interval_path`` (leaf restriction + composition,
    exact Chen relations for *any* time grid, dyadic or not).
    """
    dtype, device = latents.dtype, latents.device
    normalize = getattr(net, 'normalize_coefficients', False)
    ones = torch.ones(latents.shape[0], device=device, dtype=dtype)
    grid = torch.linspace(t0, 1.0, nfe + 1, device=device, dtype=dtype)
    state = latents
    for s_cur, t_nxt in zip(grid[:-1], grid[1:], strict=True):
        s_b = s_cur.expand_as(ones)
        t_b = t_nxt.expand_as(ones)
        path = martingale_interval_path(
            realization, schedule, s_cur, t_nxt, ei_weighted=True, zeta=zeta
        )
        delta_m = path.raw_increment()
        features = path.coefficient_features(normalize=normalize)
        z_t = ei_doss_map(net, schedule, state, s_b, t_b, features, zeta)
        state = ei_doss_reconstruct(schedule, z_t, t_b, delta_m, zeta)
    return state.to(dtype)


def main() -> None:
    """Render the requested grid from one checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path, help='Run .hydra/config.yaml')
    parser.add_argument('--output-dir', type=Path, default=Path('temporary-images'))
    parser.add_argument(
        '--mode',
        choices=('independent', 'chen', 'strong-select'),
        default='chen',
        help='Independent-path square grid, Chen-coupled NFE grid, or strong-error-'
        'selected Chen grid (top rows by strong error vs the max-NFE rollout).',
    )
    parser.add_argument('--steps', type=int, default=8, help='independent mode: flow-map steps.')
    parser.add_argument('--grid-size', type=int, default=4, help='independent mode: grid edge.')
    parser.add_argument(
        '--nfes',
        nargs='+',
        type=int,
        default=[1, 2, 4, 8, 16],
        help='chen mode: NFE per column, left (1) to right (max).',
    )
    parser.add_argument(
        '--rows',
        type=int,
        default=5,
        help='chen mode: number of (X0, path) realizations per page (grid rows).',
    )
    parser.add_argument(
        '--pages',
        type=int,
        default=1,
        help='chen mode: number of pages; each page draws fresh (X0, path) rows.',
    )
    parser.add_argument(
        '--rollouts',
        type=int,
        default=4096,
        help='strong-select mode: number of (X0, path) rollouts to score.',
    )
    parser.add_argument(
        '--chunk',
        type=int,
        default=256,
        help='strong-select mode: rollout batch size for scoring.',
    )
    parser.add_argument(
        '--top',
        type=int,
        default=4,
        help='strong-select mode: number of lowest-error rows for the final grid.',
    )
    parser.add_argument(
        '--label-id',
        type=int,
        default=None,
        help='chen mode: fixed ImageNet class id for the whole grid (e.g. 339 = horse).',
    )
    parser.add_argument('--seed', type=int, default=0, help='Seed for the shared initial latent.')
    parser.add_argument('--label-seed', type=int, default=0, help='Seed for random class labels.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, config, meta = load_student(args.config, args.checkpoint, device)
    global_step = meta['global_step']
    schedule = hydra.utils.instantiate(config.model.schedule)
    degree = int(config.model.loss.degree)
    t0 = float(config.model.loss.epsilon)
    module_cls = hydra.utils.get_class(config.model._target_)
    ei_doss = issubclass(module_cls, EIDossSussmannLitModule)
    zeta = 1.0
    anneal_steps = OmegaConf.select(config, 'model.loss.zeta_anneal_steps', default=None)
    if ei_doss and anneal_steps:
        zeta = min(1.0, global_step / float(anneal_steps))

    decoder = StabilityVAEDecoder('stabilityai/sd-vae-ft-mse')
    decoder.to(device)

    run_name = OmegaConf.select(config, 'logger.wandb.name', default='ssfm') or 'ssfm'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        if args.mode == 'independent':
            count = args.grid_size**2
            latent_generator = torch.Generator(device=device).manual_seed(args.seed)
            latents = randn((count, *LATENT_SHAPE), device=device, generator=latent_generator)
            label_generator = torch.Generator().manual_seed(args.label_seed)
            labels = torch.randint(0, net.num_classes, (count,), generator=label_generator).to(
                device
            )
            net.set_labels(labels)
            try:
                with _bf16_inference(device):
                    path_generator = torch.Generator(device=device).manual_seed(
                        args.seed + args.steps
                    )
                    if ei_doss:
                        samples = ei_doss_sussmann_sampler(
                            net,
                            schedule,
                            latents,
                            num_steps=args.steps,
                            degree=degree,
                            t0=t0,
                            zeta=zeta,
                            generator=path_generator,
                        )
                    else:
                        samples = ei_euler_sampler(
                            net,
                            schedule,
                            latents,
                            num_steps=args.steps,
                            degree=degree,
                            t0=t0,
                            generator=path_generator,
                        )
            finally:
                net.set_labels(None)
            images = decoder(samples)
            stem = f'{run_name}_step{global_step}_nfe{args.steps}_independent_paths'
            save_image(
                images.float().clamp(0.0, 1.0),
                args.output_dir / f'{stem}.png',
                nrow=args.grid_size,
            )
            print(f'Saved independent_paths grid to {args.output_dir / f"{stem}.png"}')
        elif args.mode == 'chen':
            if not ei_doss:
                raise NotImplementedError('chen mode is only implemented for EI-Doss/WZ modules.')
            if args.label_id is None:
                raise ValueError('chen mode requires --label-id (e.g. 339 = horse).')
            if args.rows < 1 or args.pages < 1:
                raise ValueError('rows and pages must be positive.')
            labels = torch.full((args.rows,), args.label_id, dtype=torch.long, device=device)
            for page in range(args.pages):
                page_seed = args.seed + 1000 * page
                latent_generator = torch.Generator(device=device).manual_seed(page_seed)
                latents = randn(
                    (args.rows, *LATENT_SHAPE), device=device, generator=latent_generator
                )
                path_generator = torch.Generator(device=device).manual_seed(page_seed + 1)
                net.set_labels(labels)
                try:
                    with _bf16_inference(device):
                        samples = chen_row_samples(
                            net,
                            schedule,
                            latents,
                            args.nfes,
                            degree,
                            t0,
                            zeta,
                            path_generator,
                            device,
                        )
                finally:
                    net.set_labels(None)
                # (nfe, path, ...) -> path-major rows so columns are NFE left -> right.
                images = decoder(samples.movedim(0, 1).flatten(0, 1))
                page_tag = f'_page{page + 1:02d}' if args.pages > 1 else ''
                stem = (
                    f'{run_name}_step{global_step}_chen_label{args.label_id}_'
                    f'nfe{"-".join(map(str, args.nfes))}{page_tag}'
                )
                save_image(
                    images.float().clamp(0.0, 1.0),
                    args.output_dir / f'{stem}.png',
                    nrow=len(args.nfes),
                )
                print(f'Saved chen grid to {args.output_dir / f"{stem}.png"}')
            print(
                f'grid layout: {args.rows} rows (independent (X0, path) realizations, top to '
                f'bottom) x {len(args.nfes)} columns (NFE {args.nfes}, left to right); label '
                f'{args.label_id}; {args.pages} page(s) under {args.output_dir}'
            )
        elif args.mode == 'strong-select':
            if not ei_doss:
                raise NotImplementedError('strong-select mode is only implemented for EI-Doss/WZ.')
            if args.label_id is None:
                raise ValueError('strong-select mode requires --label-id (e.g. 339 = horse).')
            nfes = sorted(args.nfes)
            ground = nfes[-1]
            lowers = [nfe for nfe in nfes if nfe != ground]
            if args.rollouts % args.chunk:
                raise ValueError('--rollouts must be a multiple of --chunk for exact replay.')

            # Score every rollout's strong error against the max-NFE endpoint.
            errors = torch.zeros(args.rollouts, len(lowers), dtype=torch.float64)
            labels = torch.full((args.chunk,), args.label_id, dtype=torch.long, device=device)
            for chunk_start in range(0, args.rollouts, args.chunk):
                chunk_index = chunk_start // args.chunk
                latent_generator = torch.Generator(device=device).manual_seed(
                    args.seed + 31 * chunk_index
                )
                latents = randn(
                    (args.chunk, *LATENT_SHAPE), device=device, generator=latent_generator
                )
                path_generator = torch.Generator(device=device).manual_seed(
                    args.seed + 10007 * chunk_index
                )
                realization = sample_piecewise_ei_clock(
                    schedule,
                    (args.chunk, *LATENT_SHAPE),
                    degree,
                    ground,
                    t0,
                    device=device,
                    dtype=torch.float32,
                    generator=path_generator,
                    zeta=zeta,
                )
                net.set_labels(labels)
                try:
                    with _bf16_inference(device):
                        reference = ei_doss_rollout(
                            net, schedule, latents, realization, ground, degree, t0, zeta
                        )
                        ref_norm = (
                            reference.flatten(1)
                            .norm(dim=1)
                            .clamp_min(torch.finfo(reference.dtype).tiny)
                        )
                        for column, nfe in enumerate(lowers):
                            endpoint = ei_doss_rollout(
                                net, schedule, latents, realization, nfe, degree, t0, zeta
                            )
                            errors[chunk_start : chunk_start + args.chunk, column] = (
                                ((endpoint - reference).flatten(1).norm(dim=1) / ref_norm)
                                .double()
                                .cpu()
                            )
                finally:
                    net.set_labels(None)
                print(f'  scored {chunk_start + args.chunk}/{args.rollouts}', flush=True)

            mean_error = errors.mean(dim=1)
            values, top_indices = mean_error.topk(args.top, largest=False)
            print('top rows by mean relative strong error:')
            for rank, (index, value) in enumerate(
                zip(top_indices.tolist(), values.tolist(), strict=True), 1
            ):
                print(f'  {rank}. rollout {index}: {value:.5f}')

            # Replay the winning rows deterministically and render the grid.
            winners = []
            winner_realizations = []
            winner_rows = []
            for index in top_indices.tolist():
                chunk_index, row = index // args.chunk, index % args.chunk
                latent_generator = torch.Generator(device=device).manual_seed(
                    args.seed + 31 * chunk_index
                )
                winners.append(
                    randn((args.chunk, *LATENT_SHAPE), device=device, generator=latent_generator)[
                        row
                    ]
                )
                path_generator = torch.Generator(device=device).manual_seed(
                    args.seed + 10007 * chunk_index
                )
                winner_realizations.append(
                    sample_piecewise_ei_clock(
                        schedule,
                        (args.chunk, *LATENT_SHAPE),
                        degree,
                        ground,
                        t0,
                        device=device,
                        dtype=torch.float32,
                        generator=path_generator,
                        zeta=zeta,
                    )
                )
                winner_rows.append(row)
            replay_latents = torch.stack(winners)
            # Stack per-leaf batches: coeffs is (N, B, *spatial); slice each
            # winner's row from its own realization and stack on the batch axis.
            replay_leaves = [
                BrownianPolynomial(
                    coeffs=torch.stack(
                        [
                            realization.leaves[leaf_index].coeffs[:, row]
                            for realization, row in zip(
                                winner_realizations, winner_rows, strict=True
                            )
                        ],
                        dim=1,
                    ),
                    s=winner_realizations[0].leaves[leaf_index].s,
                    t=winner_realizations[0].leaves[leaf_index].t,
                )
                for leaf_index in range(ground)
            ]
            replay = PiecewiseBrownianPath(
                torch.linspace(t0, 1.0, ground + 1, device=device),
                replay_leaves,
            )
            grid_rows = []
            net.set_labels(torch.full((args.top,), args.label_id, dtype=torch.long, device=device))
            try:
                with _bf16_inference(device):
                    for nfe in nfes:
                        grid_rows.append(
                            ei_doss_rollout(
                                net,
                                schedule,
                                replay_latents,
                                replay,
                                nfe,
                                degree,
                                t0,
                                zeta,
                            )
                        )
            finally:
                net.set_labels(None)
            images = decoder(torch.stack(grid_rows).movedim(0, 1).flatten(0, 1))
            stem = (
                f'{run_name}_step{global_step}_strongselect_label{args.label_id}_'
                f'nfe{"-".join(map(str, nfes))}_top{args.top}'
            )
            save_image(
                images.float().clamp(0.0, 1.0),
                args.output_dir / f'{stem}.png',
                nrow=len(nfes),
            )
            print(f'Saved strong-select grid to {args.output_dir / f"{stem}.png"}')
            stats = {
                'run_name': run_name,
                'checkpoint': str(args.checkpoint),
                'global_step': global_step,
                'nfes': nfes,
                'ground_truth_nfe': ground,
                'rollouts': args.rollouts,
                'chunk': args.chunk,
                'label_id': args.label_id,
                'degree': degree,
                't0': t0,
                'zeta': zeta,
                'seed': args.seed,
                'metric': 'relative L2 latent error vs max-NFE endpoint, per realization',
                'per_nfe_mean': {
                    str(nfe): float(errors[:, column].mean()) for column, nfe in enumerate(lowers)
                },
                'per_nfe_median': {
                    str(nfe): float(errors[:, column].median())
                    for column, nfe in enumerate(lowers)
                },
                'top_rows': [
                    {
                        'rollout_index': int(index),
                        'mean_error': float(mean_error[index]),
                        'per_nfe_error': {
                            str(nfe): float(errors[index, column])
                            for column, nfe in enumerate(lowers)
                        },
                    }
                    for index in top_indices.tolist()
                ],
            }
            stats_path = args.output_dir / f'{stem}_stats.json'
            stats_path.write_text(json.dumps(stats, indent=2) + '\n')
            print(f'Saved strong-error stats to {stats_path}')
        else:
            raise ValueError(f'Unknown mode {args.mode}.')

    print(
        f'run={run_name} checkpoint_step={global_step} mode={args.mode} degree={degree} '
        f't0={t0} zeta={zeta if ei_doss else None} seed={args.seed} label_seed={args.label_seed}'
    )


if __name__ == '__main__':
    main()
